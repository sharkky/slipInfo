from __future__ import annotations

import gzip as _gzip
import json
import re
from pathlib import Path  # noqa: F401
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from pyzbar.pyzbar import decode as pyzbar_decode

app = FastAPI(title="KBank Slip OCR", version="1.0.0")
app.add_middleware(GZipMiddleware, minimum_size=1000)

TYPHOON_API_KEY = "sk-Sr89gHkKQeoKVUmD3PINNUTyi73FjXCc2Z4uLORk0aCW6UxM"
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

BGRImage = npt.NDArray[np.uint8]
HSVImage = npt.NDArray[np.uint8]


# ─────────────────────────────────────────
# Image Cleaning
# ─────────────────────────────────────────
def clean_slip_image(img: BGRImage) -> BGRImage:
    gray: npt.NDArray[np.uint8] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img.astype(np.int32))
    diff_rg = np.abs(r - g)
    diff_rb = np.abs(r - b)
    diff_gb = np.abs(g - b)
    max_diff = np.maximum(np.maximum(diff_rg, diff_rb), diff_gb)

    text_mask: npt.NDArray[np.bool_] = (max_diff < 30) & (gray > 30) & (gray < 160)
    mask_u8: npt.NDArray[np.uint8] = text_mask.astype(np.uint8) * 255
    nb, output, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    result_mask = np.zeros_like(mask_u8)
    for i in range(1, nb):
        x, y, cw, ch, area = stats[i]
        if 5 <= area <= 3000 and ch < 200 and cw < 400:
            result_mask[output == i] = 255

    result: BGRImage = np.ones_like(img) * 255
    result[result_mask > 0] = [0, 0, 0]
    return result


def find_kasikorn(hsv: HSVImage) -> tuple[int, int, int] | None:
    half_w = hsv.shape[1] // 2
    hsv_left: HSVImage = hsv[:, :half_w]

    mask_red1: npt.NDArray[np.uint8] = cv2.inRange(
        hsv_left, np.array([0, 100, 100]), np.array([10, 255, 255])
    )
    mask_red2: npt.NDArray[np.uint8] = cv2.inRange(
        hsv_left, np.array([160, 100, 100]), np.array([179, 255, 255])
    )
    mask_red: npt.NDArray[np.uint8] = cv2.bitwise_or(mask_red1, mask_red2)

    best: tuple[int, int, int] | None = None
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if area / (np.pi * r * r) > 0.5:
            if best is None or r > best[2]:
                best = (int(x), int(y), int(r))
    return best


def crop_circle(img: BGRImage, cx: int, cy: int, r: int) -> npt.NDArray[np.uint8]:
    import base64  # noqa: F401

    pad = int(r * 0.1)
    r_pad = r + pad

    h, w = img.shape[:2]
    x1, y1 = max(cx - r_pad, 0), max(cy - r_pad, 0)
    x2, y2 = min(cx + r_pad, w), min(cy + r_pad, h)
    cropped: BGRImage = img[y1:y2, x1:x2].copy()

    ch, cw = cropped.shape[:2]
    mask: npt.NDArray[np.uint8] = np.zeros((ch, cw), dtype=np.uint8)
    local_cx = cx - x1
    local_cy = cy - y1
    cv2.circle(mask, (local_cx, local_cy), r, 255, -1)

    bgra: npt.NDArray[np.uint8] = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    return bgra


import cv2
import numpy as np


def find_logo_by_saturation(img, kx, ky, kr):
    """
    ใช้ตำแหน่ง X ของกสิกร (kx) แล้วสแกนลงมาข้างล่างในแนวตั้ง
    เพื่อหาวัตถุที่มีความสดของสี (Saturation) และเป็นวงกลม
    """
    h, w = img.shape[:2]

    # 1. แปลงเป็น HSV เพื่อแยกความสดของสี (Saturation)
    # วิธีนี้จะช่วยตัดลายน้ำที่เป็นสีเทาๆ หรือขาวๆ ออกไปได้ดีมาก
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]  # ดึงเฉพาะ Saturation (ความสด)

    # 2. สร้าง Mask สำหรับจุดที่มีสีสัน (ตัดส่วนที่เป็นสีขาว/เทา/ดำออก)
    # ปกติโลโก้ธนาคารจะสีสด (ม่วง, ฟ้า, ชมพู) ค่า Saturation มักจะ > 40
    _, thresh = cv2.threshold(s_channel, 40, 255, cv2.THRESH_BINARY)

    # 3. กำหนดพื้นที่ค้นหา (ROI) คือแนวคอลัมน์เดียวกับกสิกร และอยู่ใต้กสิกรลงมา
    search_width = int(kr * 2.5)
    x1 = max(0, kx - search_width // 2)
    x2 = min(w, kx + search_width // 2)
    y_start = ky + int(kr * 2)  # เริ่มหาจากใต้โลโก้กสิกรลงมา

    mask_roi = thresh[y_start:, x1:x2]
    img_roi = img[y_start:, x1:x2]

    # 4. ค้นหารูปทรงในพื้นที่ที่เราจำกัดไว้
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_circle = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # กรองขนาดให้ใกล้เคียงกับโลโก้เดิม (บวก/ลบ 40%)
        k_area = np.pi * (kr**2)
        if area < k_area * 0.4 or area > k_area * 1.6:
            continue

        (rx, ry), r = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * (r**2)

        # ตรวจสอบความเป็นวงกลม
        if area / circle_area > 0.6:
            # คืนค่าตำแหน่งจริง (ต้องบวก Offset ของ ROI กลับเข้าไป)
            real_x = int(rx + x1)
            real_y = int(ry + y_start)
            best_circle = (real_x, real_y, int(r))
            break  # เจออันแรกที่อยู่ใกล้กสิกรที่สุดก็เอาอันนั้นเลย

    return best_circle


def mask_logos_and_clean(
    img: npt.NDArray[np.uint8], shrink: int = 3
) -> tuple[bytes, str | None]:
    import base64

    # --- ส่วนเดิมของคุณ: หา Kasikorn ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kasikorn = find_kasikorn(hsv)  # ฟังก์ชันเดิมที่คุณบอกว่าแม่นแล้ว

    logo_b64 = None

    if kasikorn:
        kx, ky, kr = kasikorn
        # ปิดโลโก้กสิกร
        cv2.circle(img, (kx, ky), kr - shrink, (0, 0, 0), -1)

        # --- ส่วนที่ปรับปรุง: หาโลโก้ผู้รับ ---
        receiver = find_logo_by_saturation(img, kx, ky, kr)

        if receiver:
            rx, ry, rr = receiver
        else:
            # Fallback ถ้าหาไม่เจอจริงๆ ให้ใช้ค่ากะระยะแบบเดิม
            rx, ry, rr = kx, ky + int(img.shape[0] * 0.255), kr

        # Crop และทำ Base64
        logo_img = crop_circle(img, rx, ry, rr - shrink)
        _, logo_buf = cv2.imencode(".png", logo_img)
        logo_b64 = base64.b64encode(logo_buf.tobytes()).decode("utf-8")

        # ปิดทับโลโก้ที่สอง
        cv2.circle(img, (rx, ry), rr - shrink, (0, 0, 0), -1)

    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes(), logo_b64


# ─────────────────────────────────────────
# QR Decode + EMVCo TLV Parser
# ─────────────────────────────────────────
class TLVTag(dict[str, Any]):
    pass


def _tlv_decode(payload: str) -> list[dict[str, Any]]:
    tags: list[dict[str, Any]] = []
    idx = 0
    while idx < len(payload):
        chunk = payload[idx:]
        if len(chunk) < 4:
            break
        tag_id = chunk[:2]
        try:
            length = int(chunk[2:4])
        except ValueError:
            break
        value = chunk[4 : 4 + length]
        tags.append({"id": tag_id, "length": length, "value": value})
        idx += 4 + length
    return tags


def _parse_qr_payload(payload: str) -> dict[str, dict[str, Any]] | None:
    if not payload or len(payload) < 4:
        return None
    tags = _tlv_decode(payload)
    if not tags:
        return None
    result: dict[str, dict[str, Any]] = {}
    for tag in tags:
        entry: dict[str, Any] = {
            "id": tag["id"],
            "length": tag["length"],
            "value": tag["value"],
        }
        if len(tag["value"]) >= 4 and re.match(r"^\d{2}\d{2}", tag["value"]):
            sub = _tlv_decode(tag["value"])
            if sub:
                entry["sub_tags"] = {s["id"]: s for s in sub}
        result[tag["id"]] = entry
    return result


def decode_slip_qr(img: BGRImage) -> dict[str, str] | None:
    decoded_list = pyzbar_decode(img)
    if not decoded_list:
        gray: npt.NDArray[np.uint8] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        decoded_list = pyzbar_decode(gray)
    if not decoded_list:
        return None

    for decoded in decoded_list:
        try:
            payload = decoded.data.decode("utf-8")
        except Exception:
            continue

        tags = _parse_qr_payload(payload)
        if not tags:
            continue

        tag00 = tags.get("00")
        if not tag00 or "sub_tags" not in tag00:
            continue

        sub: dict[str, dict[str, Any]] = tag00["sub_tags"]
        api_type: str = sub.get("00", {}).get("value", "")
        sending_bank: str = sub.get("01", {}).get("value", "")
        trans_ref: str = sub.get("02", {}).get("value", "")

        if api_type == "000001" and sending_bank and trans_ref:
            return {"sendingBank": sending_bank, "transRef": trans_ref}

    return None


# ─────────────────────────────────────────
# OCR
# ─────────────────────────────────────────
def extract_slip_data(image_bytes: bytes) -> str | None:
    req = requests.post(
        "http://localhost:5007/ocr",
        files={"image": ("slip.jpg", image_bytes, "image/jpeg")},
    )
    if req.status_code != 200:
        return None
    return req.json().get("text", None).replace("↓", "").strip()


import re
from typing import Any

# ─────────────────────────────────────────
# Patterns & Parsers
# ─────────────────────────────────────────

TYPE_MAP: dict[str, str] = {
    "โอนเงินสำเร็จ": "normal",
    "ชำระบิลสำเร็จ": "bill",
    "ชำระบิล": "bill",
    "ชำระเงินสำเร็จ": "bill",
    "ชำระสำเร็จ": "bill",
    "ชำระเงิน": "bill",
    "เติมเงินสำเร็จ": "bill",
    "จ่ายบิลสำเร็จ": "bill",
}

SENDER_ACCOUNT_RE = re.compile(r"^x[\dx\-\s]{4,}[\dx]$", re.IGNORECASE)

PROMPTPAY_PHONE_RE = re.compile(r"^x{2,3}-x{2,3}-\d{1,4}$")
PROMPTPAY_ID_RE = re.compile(r"^\d{2,3}-x{3,}-\d{1,4}$")

PROMPTPAY_LABEL_RE = re.compile(
    r"^(Prompt\s*Pay|พร้อมเพย์|PromptPay|รหัสพร้อมเพย์|Mobile|QR\s*Code|QR)$",
    re.IGNORECASE,
)

REF_RE = re.compile(r"^(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*[0-9])[A-Z0-9]{15,}$")

REF_SEARCH_RE = re.compile(
    r"(?<![A-Z0-9])(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*[0-9])[A-Z0-9]{15,}(?![A-Z0-9])"
)

DATE_RE = re.compile(
    r"\d{1,2}\s+[ก-ๆ็-๎]+\.?\s*[ก-ๆ็-๎]*\.?\s+\d{2,4}\s+\d{1,2}:\d{2}\s*น\.?"
)

HEADER_NOISE_RE = re.compile(
    r"(โอนเงิน|ชำระบิล|ชำระเงิน|ชำระ|สำเร็จ|K\+|TRUE\s*MONEY|SCB\s*EASY|"
    r"น\.\s*$|มี\.ค\.|ม\.ค\.|ก\.พ\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|"
    r"ต\.ค\.|พ\.ย\.|ธ\.ค\.|^\d{1,2}:\d{2})",
    re.IGNORECASE,
)

SEPARATOR_RE = re.compile(r"^[↓→←↑\-–—|/\\]+$")

SINGLE_LETTER_ID_RE = re.compile(r"^[A-Z]\d{10,}$")

KBANK_REF_RE = re.compile(r"^0\d+[A-Z]{2,}[A-Z0-9]+$")

TAIL_TRIGGERS: list[str] = [
    "เลขที่รายการ",
    "เลขอ้างอิง",
    "จำนวน",
    "ค่าธรรมเนียม",
    "ค่าบริการ",
    "<figure>",
    "สแกน",
]


class PartyInfo(dict[str, str]):
    pass


SlipResult = dict[str, Any]


# ─────────────────────────────────────────
# Utils
# ─────────────────────────────────────────


def is_sender_account(line: str) -> bool:
    return bool(SENDER_ACCOUNT_RE.match(line.strip()))


def is_tail(line: str) -> bool:
    s = line.strip()

    if SINGLE_LETTER_ID_RE.match(s):
        return False

    return any(t in line for t in TAIL_TRIGGERS) or bool(KBANK_REF_RE.match(s))


def is_noise(line: str) -> bool:
    return (
        bool(HEADER_NOISE_RE.search(line))
        or bool(SEPARATOR_RE.match(line.strip()))
        or bool(DATE_RE.search(line))
    )


def parse_amount(text: str) -> float:
    m = re.search(r"[\d,]+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group().replace(",", "")) if m else 0.0


# def normalize_account(acc: str) -> str:
#     acc = re.sub(r"(?<=[xX\d])\s+(?=[xX\d])", "-", acc)
#     return acc.lower()
def normalize_account(acc: str) -> str:

    acc = acc.strip()

    # normalize dash spacing
    acc = re.sub(r"\s*-\s*", "-", acc)

    # lowercase เฉพาะ X mask
    acc = re.sub(r"[X]", "x", acc)

    return acc


# ─────────────────────────────────────────
# Sender Parser
# ─────────────────────────────────────────


def _parse_sender_lines(lines: list[str]) -> dict[str, str]:

    if not lines:
        return {"name": "", "bank": "", "account": ""}

    if len(lines) == 1:
        return {"name": "", "bank": "", "account": lines[0]}

    account = normalize_account(lines[-1])

    bank = lines[-2] if len(lines) >= 2 else ""

    name = " ".join(lines[:-2]) if len(lines) >= 3 else ""

    return {
        "name": name.strip(),
        "bank": bank.strip(),
        "account": account,
    }


# ─────────────────────────────────────────
# Receiver Parser (FIXED)
# ─────────────────────────────────────────


def _parse_receiver_lines(lines: list[str]) -> dict[str, str]:

    if not lines:
        return {"name": "", "bank": "", "account": ""}

    if len(lines) == 1:
        return {"name": "", "bank": "", "account": lines[0].strip()}

    if len(lines) == 2:
        return {
            "name": lines[0].strip(),
            "bank": normalize_account(lines[1].strip()),
            "account": "",
        }

    # ≥3 lines → positional parsing
    name = lines[0].strip()
    bank = lines[1].strip()
    account = lines[2].strip()

    return {
        "name": name,
        "bank": normalize_account(bank),
        "account": normalize_account(account),
    }


# ─────────────────────────────────────────
# Empty Result
# ─────────────────────────────────────────


def _empty(slip_type: str, date: str) -> SlipResult:

    ep = {"name": "", "bank": "", "account": ""}

    return {
        "type": slip_type,
        "date": date,
        "sender": ep,
        "receiver": ep,
        "ref": "",
        "amount": 0.0,
        "fee": 0.0,
    }


# ─────────────────────────────────────────
# Main Parser
# ─────────────────────────────────────────


def parse_slip_to_json(raw_text: str) -> SlipResult:

    lines = [
        l.strip()
        for l in raw_text.splitlines()
        if l.strip() and not SEPARATOR_RE.match(l.strip())
    ]

    # ── type

    slip_type = "unknown"

    for line in lines:
        for kw, val in TYPE_MAP.items():
            if kw in line:
                slip_type = val
                break
        if slip_type != "unknown":
            break

    # ── date

    date = ""

    for line in lines:
        m = DATE_RE.search(line)
        if m:
            date = m.group().strip()
            break

    # ── sender

    first_account_idx = next(
        (i for i, l in enumerate(lines) if is_sender_account(l)), None
    )

    if first_account_idx is None:
        return _empty(slip_type, date)

    sender_start = first_account_idx

    for i in range(first_account_idx - 1, -1, -1):
        if is_noise(lines[i]):
            sender_start = i + 1
            break

        if i == 0:
            sender_start = 0

    sender = _parse_sender_lines(lines[sender_start : first_account_idx + 1])

    # ── receiver

    tail_idx = len(lines)

    for i in range(first_account_idx + 1, len(lines)):
        if i <= first_account_idx + 3:
            continue

        if is_tail(lines[i]):
            tail_idx = i
            break

    receiver_lines = []

    for l in lines[first_account_idx + 1 :]:
        if any(t in l for t in TAIL_TRIGGERS):
            break

        receiver_lines.append(l)

    receiver = _parse_receiver_lines(receiver_lines)

    # ── tail

    tail = lines[tail_idx:]

    ref = ""
    amount = 0.0
    fee = 0.0

    i = 0

    while i < len(tail):
        line = tail[i]

        if not ref and ("เลขที่รายการ" in line or "เลขอ้างอิง" in line):
            m = REF_SEARCH_RE.search(line)

            if m:
                ref = m.group()

            else:
                nxt = tail[i + 1].strip() if i + 1 < len(tail) else ""
                ref = nxt if REF_RE.match(nxt) else ""

        if not ref and REF_RE.match(line.strip()):
            ref = line.strip()

        if "จำนวน" in line:
            m2 = re.search(r"([\d,]+(?:\.\d+)?)\s*บาท", line)

            amount = (
                parse_amount(m2.group(1))
                if m2
                else (parse_amount(tail[i + 1]) if i + 1 < len(tail) else 0.0)
            )

        if "ค่าธรรมเนียม" in line or "ค่าบริการ" in line:
            m3 = re.search(r"([\d,]+(?:\.\d+)?)\s*บาท", line)

            fee = (
                parse_amount(m3.group(1))
                if m3
                else (parse_amount(tail[i + 1]) if i + 1 < len(tail) else 0.0)
            )

        i += 1

    return {
        "type": slip_type,
        "date": date,
        "sender": sender,
        "receiver": receiver,
        "ref": ref,
        "amount": amount,
        "fee": fee,
    }


def compress_jpeg_bgr(img: BGRImage, quality: int = 80) -> BGRImage:
    success, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])

    if not success:
        raise ValueError("JPEG compression failed")

    compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if compressed is None:
        raise ValueError("JPEG decode failed")

    return compressed


@app.post("/ocr-receipt")
async def ocr_receipt(image: UploadFile = File(...)) -> Response:
    if image.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{image.content_type}'. Allowed: {', '.join(SUPPORTED_CONTENT_TYPES)}",
        )

    raw_bytes = await image.read()

    img_array: npt.NDArray[np.uint8] = np.frombuffer(raw_bytes, dtype=np.uint8)
    img: BGRImage = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="ไม่สามารถอ่านไฟล์รูปได้")

    # compress image
    img = compress_jpeg_bgr(img, quality=96)

    qr_info = decode_slip_qr(img.copy())

    if qr_info is None:
        raise HTTPException(
            status_code=422,
            detail="ไม่พบ QR Code ในรูปนี้ หรือ QR Code ไม่ใช่สลิปโอนเงิน",
        )

    sending_bank = qr_info.get("sendingBank", "")
    if sending_bank != "004":
        raise HTTPException(
            status_code=422,
            detail=f"สลิปนี้ไม่ใช่ของ KBank (รหัสธนาคาร: '{sending_bank}') ระบบรองรับเฉพาะ KBank (004) เท่านั้น",
        )

    cleaned_bytes, logo_b64 = mask_logos_and_clean(img.copy(), shrink=3)

    with open("cleaned_slip.jpg", "wb") as f:
        f.write(cleaned_bytes)

    raw_text = extract_slip_data(cleaned_bytes)
    print(raw_text)
    if not raw_text:
        raise HTTPException(status_code=502, detail="OCR ไม่สามารถอ่านข้อความจากรูปได้")

    result: SlipResult = parse_slip_to_json(raw_text)

    if result["type"] == "unknown":
        raise HTTPException(status_code=422, detail="ไม่สามารถระบุประเภทสลิปได้")

    if qr_info.get("transRef"):
        result["ref"] = qr_info["transRef"]

    result["receiver"]["logo"] = logo_b64
    result["ocr"] = raw_text

    body = json.dumps(result, ensure_ascii=False).encode("utf-8")
    compressed = _gzip.compress(body, compresslevel=6)
    return Response(
        content=compressed,
        media_type="application/json",
        headers={"Content-Encoding": "gzip"},
    )
