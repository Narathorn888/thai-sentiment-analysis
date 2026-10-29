import re

# Domain keywords for E-Commerce CRM Issue Classification
ISSUE_CATEGORIES = {
    "การจัดส่งและขนส่ง (Delivery & Logistics)": [
        "ส่ง", "ส่งช้า", "ส่งไว", "ขนส่ง", "พัสดุ", "Kerry", "Flash", "J&T", "DHL", 
        "ไปรษณีย์", "นาน", "ถึงช้า", "กล่องบุบ", "แตกหัก", "โยน", "ส่งผิด", "พนักงานส่ง", "ไรเดอร์"
    ],
    "คุณภาพสินค้าและความถูกต้อง (Product Quality)": [
        "พัง", "ชำรุด", "เสียหาย", "ใช้ไม่ได้", "ของปลอม", "ไม่ตรงปก", "หมดอายุ", 
        "คุณภาพ", "บอบบาง", "ขาด", "ของแท้", "เกรดดี", "มีรอย", "ของไม่ครบ", "ผิดสี", "ผิดไซส์"
    ],
    "การบริการและการติดต่อแอดมิน (Customer Support)": [
        "แอดมิน", "ติดต่อ", "ตอบช้า", "ไม่ตอบ", "พนักงาน", "บริการ", "พูดจา", 
        "ไม่รับผิดชอบ", "โทร", "ถาม", "ช่วย", "เมิน", "บล็อก", "ศูนย์บริการ"
    ],
    "ราคา คูปอง และการคืนเงิน (Pricing & Refund)": [
        "แพง", "คุ้ม", "ราคา", "โค้ด", "คูปอง", "คืนเงิน", "ยกเลิก", "ค่าส่ง", 
        "แพงเกิน", "ใบเสร็จ", "ยอดเงิน", "หักเงิน", "เงินคืน", "จ่ายเงิน"
    ]
}

def classify_crm_issue(text: str) -> tuple:
    """
    Identifies the primary issue category mentioned in the customer review.
    Returns: (primary_category, matched_keywords)
    """
    if not isinstance(text, str):
        return "ข้อคิดเห็นทั่วไป (General / Other)", []
    
    text_lower = text.lower()
    score_dict = {}
    matched_words_dict = {}
    
    for category, keywords in ISSUE_CATEGORIES.items():
        matched = [kw for kw in keywords if kw.lower() in text_lower]
        if matched:
            score_dict[category] = len(matched)
            matched_words_dict[category] = matched
            
    if not score_dict:
        return "ข้อคิดเห็นทั่วไป (General / Other)", []
        
    best_category = max(score_dict, key=score_dict.get)
    return best_category, matched_words_dict[best_category]

def determine_crm_priority(sentiment: str, issue_category: str) -> dict:
    """
    Assigns an operational priority level for the CRM support team.
    """
    if sentiment == "เชิงลบ":
        if "Product Quality" in issue_category or "Customer Support" in issue_category:
            return {
                "level": "🔴 วิกฤต / ด่วนที่สุด (Critical)",
                "action": "ติดต่อลูกค้าทันทีภายใน 1 ชม. เพื่อตรวจสอบการเคลมหรือขออภัย",
                "color": "#EF4444"
            }
        else:
            return {
                "level": "🟠 เร่งด่วน (High)",
                "action": "ตรวจสอบสถานะการขนส่งหรือการคืนเงินภายใน 4 ชม.",
                "color": "#F97316"
            }
    elif sentiment == "เชิงกลาง":
        return {
            "level": "🟡 ปกติ (Medium)",
            "action": "จัดเก็บเข้าสู่ระบบติดตามความพึงพอใจทั่วไป",
            "color": "#EAB308"
        }
    else:  # เชิงบวก
        return {
            "level": "🟢 ความพึงพอใจสูง (Low Priority)",
            "action": "ส่งข้อความขอบคุณ หรือแนะนำแคมเปญโปรโมชั่นสำหรับลูกค้าประจำ",
            "color": "#22C55E"
        }

