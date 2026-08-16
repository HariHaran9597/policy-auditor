"""Generate the synthetic demo corpus: 3 Acme Corp policy PDFs with reportlab.

The documents are deliberately dense in tables and hard numbers (grades,
limits, SLAs) — exactly the content where naive RAG hallucinates and where
table-aware parsing + hybrid retrieval + reranking earn their keep.

Run:  python scripts/make_sample_data.py [--out data/policies]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BODY = ParagraphStyle(
    "Body", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=8
)
H1 = ParagraphStyle(
    "H1x", fontName="Helvetica-Bold", fontSize=14, leading=18, spaceBefore=14, spaceAfter=8
)
TITLE = ParagraphStyle(
    "Tx", fontName="Helvetica-Bold", fontSize=20, leading=26, spaceAfter=10, alignment=1
)
SUB = ParagraphStyle(
    "Sub", fontName="Helvetica", fontSize=11, leading=15, spaceAfter=4, alignment=1
)

GRID = TableStyle(
    [
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
)


CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=9, leading=11)
CELLH = ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=9, leading=11)


def _table(header: list[str], rows: list[list[str]], col_widths: list[float] | None = None):
    # Paragraph cells (not plain strings) so text wraps INSIDE the cell borders —
    # plain strings overflow the column and interleave with the next column.
    data = [[Paragraph(h, CELLH) for h in header]] + [
        [Paragraph(c, CELL) for c in row] for row in rows
    ]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(GRID)
    return t


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2 * cm, 1.2 * cm, "Acme Corp — Internal / Confidential")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _build(path: Path, title: str, subtitle: str, story):
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title=title,
    )
    full = [Paragraph(title, TITLE), Paragraph(subtitle, SUB), Spacer(1, 10)] + story
    doc.build(full, onFirstPage=_footer, onLaterPages=_footer)


# ----------------------------------------------------------------------------
# Document 1: Employee Handbook
# ----------------------------------------------------------------------------

HANDBOOK_SECTIONS: list = [
    ("1. Introduction and Scope", [
        "This Employee Handbook (the Handbook) is issued by Acme Corp India Private Limited (Acme Corp) and is effective from 1 January 2026. It supersedes all previous handbooks, circulars, and verbal guidance on the matters covered herein.",
        "The Handbook applies to all employees of Acme Corp on the India payroll, including full-time, part-time, and fixed-term employees. It does not apply to independent contractors, consultants engaged through vendors, or interns (who are governed by the separate Internship Policy).",
        "This document is classified Internal. Questions about interpretation must be raised with the HR Business Partner (HRBP) assigned to your business unit, or through the HR portal at hr.acme.example.",
    ]),
    ("2. Employment Classification and Probation", [
        "Every new regular employee is placed on probation for a period of six (6) months from the date of joining. Probation may be extended once, by up to three (3) additional months, with written reasons provided to the employee.",
        "Confirmation of employment requires a probation review rating of Meets Expectations or higher, completion of all mandatory onboarding trainings, and return of all signed policy acknowledgements to HR.",
        "Employment classifications used throughout this Handbook: Grade L1 (Associate), Grade L2 (Senior Associate), Grade L3 (Manager and above), Grade L4 (Director and above). Where a rule differs by grade, the grade recorded in the HR portal on the date the rule applies is authoritative.",
    ]),
    ("3. Working Hours and Attendance", [
        "The standard work week is 45 hours, Monday to Friday. Core hours during which every office-based employee must be reachable are 10:00 to 16:00 India Standard Time; flexible band hours are 07:00 to 19:00.",
        "Employees working more than 9 hours in a day are entitled to a 45-minute paid break in addition to the standard 30-minute lunch break.",
        "Late attendance is recorded when an employee arrives after the end of their declared flexible band start time plus a grace period of 15 minutes. More than three (3) recorded late arrivals in a calendar month trigger a documented verbal warning from the reporting manager; six or more trigger a written warning.",
        "Overtime compensation applies only to Grade L1 employees, at 1.5x hourly basic pay, and must be pre-approved in writing by the department head.",
    ]),
    ("4. Leave Policy", [
        "The leave year runs from 1 January to 31 December. All leave must be applied for in the HR portal before it is taken, except in emergencies covered by sick leave rules below.",
        "The following leave types and annual entitlements apply to all confirmed employees:",
        "TABLE_LEAVE",
        "Leave entitlements are pro-rated in the joining year and in the year of exit.",
        "Carry-over: up to 8 days of unused annual leave may be carried into the next leave year; carried-over leave expires on 31 March of that year. Sick leave and casual leave cannot be carried over under any circumstances.",
        "Annual leave of 3 or more consecutive working days requires 14 calendar days of advance notice and manager approval recorded in the HR portal. Leave taken without approval is treated as loss of pay.",
        "Sick leave of more than 3 consecutive working days must be supported by a registered medical practitioner's certificate submitted within 5 working days of rejoining.",
        "Maternity leave of 26 weeks applies as per the Maternity Benefit Act, of which not more than 8 weeks may precede the expected delivery date. Paternity leave of 4 weeks must be availed within 6 months of the child's birth, in blocks of no less than 1 week.",
        "Bereavement leave of 5 working days applies to the loss of an immediate family member (spouse, child, parent, sibling, or grandparent) and must be applied retroactively within 7 working days.",
    ]),
    ("5. Remote and Hybrid Work", [
        "Employees whose registered work location is within 50 km of an Acme Corp office must work from that office at least two (2) days per calendar week, unless an exception is approved in writing by the department Director.",
        "Fully remote status (fewer than one office day per month) requires approval of the department Director and the Chief People Officer, is reviewed every six (6) months, and may be revoked with 30 days notice.",
        "A remote work internet allowance is paid to approved fully-remote employees: 1,000 INR per month for Grades L1 to L3. See also Addendum A (2026) for the revised allowance effective 1 March 2026.",
        "Employees working remotely from a location other than their registered address for more than 30 consecutive days must inform HR in writing, as tax and statutory obligations may change.",
        "All remote employees must comply with the Acme Corp IT Security Policy, including mandatory disk encryption and VPN usage for all company systems.",
    ]),
    ("6. Code of Conduct", [
        "Employees must disclose any outside employment, directorship, or financial interest greater than 5 percent in a vendor, customer, or competitor of Acme Corp within 10 working days of it arising, using the Conflict of Interest form on the HR portal.",
        "Gifts or hospitality from business partners with a value above 2,500 INR must be declined or declared to Compliance within 5 working days. Cash or cash-equivalent gifts are prohibited in all cases.",
        "Side employment (secondary occupation) requires prior written approval from the reporting manager and the HRBP; approval is denied wherever the secondary occupation competes with Acme Corp or exceeds 20 hours per week.",
        "Public statements about Acme Corp, including on social media, must follow the Communications Policy; employees must not disclose Confidential or Restricted information as defined in the IT Security Policy.",
        "Violations of the Code of Conduct are handled under the disciplinary matrix in Section 12; gross misconduct may result in termination without notice.",
    ]),
    ("7. Anti-Harassment and Grievance Redressal", [
        "Acme Corp maintains a zero-tolerance position on workplace harassment, including sexual harassment, which is administered by the Internal Committee (IC) constituted under the POSH Act.",
        "Complaints may be raised with any of the following, in any order: the immediate manager, the HR Business Partner, any member of the Internal Committee, or the anonymous ethics hotline (ethics.acme.example). Complaints are acknowledged within 2 working days.",
        "The grievance resolution timeline is as follows, with escalation automatic when a stage exceeds its limit:",
        "TABLE_GRIEVANCE",
        "POS complaints are inquiry-completed within 90 days as required by law, and the IC's recommendations are implemented within 60 days of the report.",
        "Retaliation against any complainant or witness is itself gross misconduct and is investigated by the Chief People Officer directly.",
    ]),
    ("8. Expense and Travel Policy", [
        "All business expenses must be genuine, business-related, supported by original invoices, and submitted through the expense portal within 30 calendar days of the expense date. Late submissions beyond 60 days are reimbursed only with CFO approval.",
        "Travel class and hotel ceilings by grade are as follows (per night, within India):",
        "TABLE_TRAVEL",
        "The meal allowance on business travel within India is 600 INR per day (no receipts required). Incidental expenses are covered at 100 INR per day.",
        "Any single expense above 25,000 INR requires pre-approval from the Chief Financial Officer in addition to the normal manager approval.",
        "International travel requires approval from the department Director, booking through the approved travel desk, and a per-diem as per the international annexure published on the finance portal.",
        "Advances must be settled within 15 days of trip completion; unspent advances are recovered from the next payroll cycle.",
    ]),
    ("9. Performance Management", [
        "The performance year runs from 1 April to 31 March. Every employee has documented quarterly check-ins with their manager in the performance portal, and a formal annual review in April.",
        "Ratings are: Exceptional, Exceeds Expectations, Meets Expectations, Needs Improvement, and Unsatisfactory. A rating of Needs Improvement places the employee on a Performance Improvement Plan (PIP) of up to 60 days with fortnightly reviews.",
        "Promotions are effective in the April and October cycles. Eligibility requires a rating of Meets Expectations or higher in the most recent cycle, completion of 12 months in the current grade, and a current-year promotion budget confirmation from Finance.",
        "Two consecutive Unsatisfactory ratings, or failure to exit a PIP successfully, may lead to separation under Section 11.",
    ]),
    ("10. Benefits Summary", [
        "All regular employees are covered by group health insurance with a family floater sum insured of 500,000 INR. Employees at Grade L3 and above receive a top-up cover of 200,000 INR. New joiners are enrolled automatically on their start date.",
        "An annual preventive health check-up is covered for employees aged 40 and above, and for all employees at Grade L3 and above, up to 6,000 INR per year.",
        "The learning and development budget is 30,000 INR per year for Grades L2 and above, usable for courses, certifications, and books with manager approval; it does not carry over.",
        "Group term life insurance of 24 months of fixed pay and personal accident cover of 12 months of fixed pay apply to all regular employees at no cost.",
        "Statutory benefits (provident fund, gratuity, and insurance under the ESI Act where applicable) are administered per law and summarized on each payslip.",
    ]),
    ("11. Separation and Exit", [
        "Resignation must be submitted through the HR portal. The notice period is 30 calendar days for Grades L1 and L2, and 60 calendar days for Grades L3 and above. Notice pay in lieu is permitted only with mutual written agreement.",
        "During the notice period the employee is expected to complete a structured handover documented in the handover template; the relieving letter is issued only after the reporting manager signs off the handover.",
        "Full and final settlement, including reimbursements and leave encashment (annual leave only, capped at 45 days), is processed within 45 calendar days of the last working day.",
        "All company assets (laptop, access cards, and peripherals) must be returned on or before the last working day; unreturned assets are recovered at book value from the settlement amount.",
        "Termination for cause follows the disciplinary matrix in Section 12 and, where applicable, the procedures of the Internal Committee under the POSH Act.",
    ]),
    ("12. Disciplinary Matrix", [
        "Disciplinary action is applied in proportion to severity, with the following standard ladder:",
        "TABLE_DISCIPLINE",
        "The reporting manager may issue a documented verbal warning; every later stage requires HRBP review and, from suspension onward, approval of the Chief People Officer.",
        "An employee may appeal any disciplinary action to the Chief People Officer within 10 working days of being informed; the appeal decision is final.",
    ]),
    ("13. Data Handling Cross-Reference", [
        "All employee and customer personal data referenced in this Handbook is classified and handled according to the Acme Corp IT Security Policy 2026, including the data classification scheme (Public, Internal, Confidential, Restricted) defined there.",
        "HR records are Confidential; identity documents, salary information, and health records are Restricted and may be accessed only by named HR roles listed in the HR data register.",
        "Suspected data breaches involving HR records must be reported to security@acme.example within 24 hours under the incident response procedure of the IT Security Policy.",
    ]),
]

LEAVE_TABLE = (
    ["Leave type", "Entitlement per year", "Advance notice", "Carry-over"],
    [
        ["Annual / earned leave", "24 working days", "14 calendar days for 3+ day blocks", "Up to 8 days, expires 31 March"],
        ["Sick leave", "12 working days", "Inform manager same day; certificate if 3+ days", "Not allowed"],
        ["Casual leave", "6 working days", "2 working days", "Not allowed"],
        ["Maternity leave", "26 weeks", "As per Maternity Benefit Act; max 8 weeks prenatal", "Not applicable"],
        ["Paternity leave", "4 weeks", "7 calendar days", "Not applicable; within 6 months of birth"],
        ["Bereavement leave", "5 working days", "Retroactive application within 7 working days", "Not applicable"],
    ],
    [3.8 * cm, 4.0 * cm, 5.2 * cm, 3.8 * cm],
)

GRIEVANCE_TABLE = (
    ["Escalation level", "Authority", "Response limit"],
    [
        ["Level 1", "Immediate manager", "3 business days"],
        ["Level 2", "HR Business Partner", "5 business days"],
        ["Level 3", "Chief People Officer", "10 business days"],
        ["POSH complaints", "Internal Committee", "Acknowledgement 2 days; inquiry within 90 days"],
    ],
    [3.8 * cm, 4.6 * cm, 8.4 * cm],
)

TRAVEL_TABLE = (
    ["Grade", "Flight class", "Hotel ceiling (per night, India)", "Local travel"],
    [
        ["L1", "Economy", "4,000 INR", "Cab, actuals"],
        ["L2", "Premium economy", "7,000 INR", "Cab, actuals"],
        ["L3", "Business", "12,000 INR", "Chauffeur for client meetings"],
        ["L4 (Director+)", "Business", "18,000 INR", "Chauffeur"],
    ],
    [3.0 * cm, 3.8 * cm, 5.6 * cm, 4.4 * cm],
)

DISCIPLINE_TABLE = (
    ["Level", "Action", "Decision authority"],
    [
        ["Stage 1", "Documented verbal warning", "Reporting manager"],
        ["Stage 2", "Written warning", "Reporting manager + HRBP"],
        ["Stage 3", "Final written warning or suspension up to 10 days", "HRBP + Chief People Officer"],
        ["Stage 4", "Termination", "Chief People Officer"],
    ],
    [2.4 * cm, 7.6 * cm, 6.8 * cm],
)

HANDBOOK_TABLES = {
    "TABLE_LEAVE": LEAVE_TABLE,
    "TABLE_GRIEVANCE": GRIEVANCE_TABLE,
    "TABLE_TRAVEL": TRAVEL_TABLE,
    "TABLE_DISCIPLINE": DISCIPLINE_TABLE,
}


# ----------------------------------------------------------------------------
# Document 2: IT Security Policy
# ----------------------------------------------------------------------------

SECURITY_SECTIONS: list = [
    ("1. Purpose and Scope", [
        "This policy defines the minimum security controls for all Acme Corp information systems, applications, and users. It applies to employees, contractors, and third parties accessing Acme Corp systems from any device or location.",
        "This policy is effective from 1 January 2026 and is reviewed annually by the Information Security Committee. Exceptions require written approval from the Chief Information Security Officer (CISO) and expire automatically after 90 days unless renewed.",
    ]),
    ("2. Password and Authentication Policy", [
        "All corporate and cloud application accounts must satisfy the following minimum requirements:",
        "TABLE_PASSWORD",
        "Multi-factor authentication (MFA) is mandatory for all cloud applications, remote VPN access, and administrator accounts, using the approved authenticator app; SMS-based MFA is not permitted for Restricted systems.",
        "Credentials must never be shared, embedded in code, or stored outside the corporate password manager. Any suspected credential exposure must be reported within 24 hours and triggers a forced reset.",
    ]),
    ("3. Device Security", [
        "Full-disk encryption (FileVault or BitLocker) is mandatory on all laptops and mobile devices used for Acme Corp work, including personal devices under the BYOD programme.",
        "Devices must auto-lock after 5 minutes of inactivity with a password or biometric unlock. Critical OS and browser patches must be applied within 72 hours of release; all other security patches within 14 days.",
        "Loss or theft of any device with access to corporate systems must be reported to security@acme.example and the IT helpdesk within 24 hours; the device is remotely wiped and all stored sessions revoked.",
        "Personal devices may access Internal-classification resources only through the managed browser; Confidential and Restricted resources require a corporate device.",
    ]),
    ("4. Data Classification", [
        "All data must be classified at creation into one of four levels:",
        "TABLE_CLASSIFICATION",
        "Personally identifiable information (PII) of employees and customers is Restricted by default. Restricted data must be encrypted at rest and in transit, and shared only through the approved secure-share portal — never as email attachments.",
        "Public disclosures of any data require sign-off from Corporate Communications and, for financial data, the CFO.",
    ]),
    ("5. Incident Response and SLAs", [
        "Security incidents must be reported to security@acme.example immediately upon suspicion. The on-call security engineer triages every report against the following severity matrix and response targets:",
        "TABLE_INCIDENT",
        "A suspected breach of personal data must additionally be notified to the Data Protection Officer within 24 hours for statutory assessment, and to affected customers per contractual terms.",
        "Every Sev1 and Sev2 incident requires a blameless post-incident review within 5 business days, with tracked corrective actions owned by the responsible engineering lead.",
    ]),
    ("6. Access Control", [
        "Access follows least privilege and role-based access control. Access requests are raised in the identity portal, approved by the system owner, and granted within 2 business days for standard roles.",
        "Access rights are reviewed quarterly by system owners; unused accounts are disabled after 45 days of inactivity and deleted after 90 days.",
        "Leavers lose all access on their last working day (same-day revocation by IT upon HR trigger). Privileged (administrator) access requires a named individual, a justification, and automatic expiry after 90 days.",
    ]),
    ("7. Third-Party and Cloud Security", [
        "Every vendor handling Acme Corp data must pass a security assessment before onboarding and annually thereafter, and must sign the standard Data Processing Agreement (DPA).",
        "Production data must never be copied into development or test environments; masked or synthetic data sets are provided by the data platform team on request.",
        "Cloud workloads must run in approved accounts with mandatory logging, and internet-exposed services require a security review plus CISO-approved exception register entry.",
    ]),
]

PASSWORD_TABLE = (
    ["Requirement", "Minimum standard"],
    [
        ["Password length", "12 characters minimum"],
        ["Complexity", "At least one uppercase, one lowercase, one digit, one special character"],
        ["Rotation", "Every 180 days; immediate change on suspicion of compromise"],
        ["History", "Last 10 passwords cannot be reused"],
        ["Account lockout", "After 5 failed attempts, locked for 15 minutes"],
        ["Password manager", "Mandatory for all corporate credentials; provided by IT"],
    ],
    [5.0 * cm, 11.8 * cm],
)

CLASSIFICATION_TABLE = (
    ["Level", "Definition", "Examples", "Handling rule"],
    [
        ["Public", "Approved for public release", "Marketing material, job postings", "No restriction"],
        ["Internal", "Default for business data", "This policy, org charts", "Employees and approved contractors"],
        ["Confidential", "Damage to Acme Corp if disclosed", "Contracts, financials, HR records", "Need-to-know; encrypted sharing"],
        ["Restricted", "Severe legal/financial harm if disclosed", "Customer PII, health records, credentials", "Named-access only; secure-share portal"],
    ],
    [2.6 * cm, 4.5 * cm, 4.4 * cm, 5.3 * cm],
)

INCIDENT_TABLE = (
    ["Severity", "Definition", "Acknowledge within", "Resolve within"],
    [
        ["Sev1", "Critical: production down or confirmed data breach", "15 minutes", "4 hours"],
        ["Sev2", "High: major function degraded, limited breach risk", "30 minutes", "8 hours"],
        ["Sev3", "Medium: single-system issue, workaround exists", "4 hours", "3 business days"],
        ["Sev4", "Low: query or cosmetic defect", "1 business day", "10 business days"],
    ],
    [1.8 * cm, 6.6 * cm, 4.0 * cm, 4.4 * cm],
)

SECURITY_TABLES = {
    "TABLE_PASSWORD": PASSWORD_TABLE,
    "TABLE_CLASSIFICATION": CLASSIFICATION_TABLE,
    "TABLE_INCIDENT": INCIDENT_TABLE,
}


# ----------------------------------------------------------------------------
# Document 3: Remote Work Addendum
# ----------------------------------------------------------------------------

ADDENDUM_SECTIONS: list = [
    ("Addendum A: Remote Work Allowance Revision (2026)", [
        "This addendum amends Section 5 of the Employee Handbook 2026 (Remote and Hybrid Work) with effect from 1 March 2026.",
        "The monthly remote work internet allowance for approved fully-remote employees is revised from 1,000 INR to 1,200 INR for Grades L1 and L2. For Grade L3 the allowance remains 1,000 INR.",
        "The allowance is payable with monthly payroll upon confirmation of fully-remote status in the HR portal and is subject to statutory deductions where applicable.",
        "All other terms of Section 5 of the Handbook, including the office-attendance requirement of two days per week for employees within 50 km of an office, remain unchanged.",
    ]),
]


def _story(sections, tables):
    story = []
    for heading, blocks in sections:
        story.append(Paragraph(heading, H1))
        for block in blocks:
            if isinstance(block, str) and block.startswith("TABLE_"):
                header, rows, widths = tables[block]
                story.append(_table(header, rows, widths))
                story.append(Spacer(1, 8))
            else:
                story.append(Paragraph(block, BODY))
        story.append(PageBreak())
    return story


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _build(
        out_dir / "Acme_Employee_Handbook_2026.pdf",
        "Acme Corp Employee Handbook 2026",
        "Version 3.2 — Effective 1 January 2026 — Owner: Chief People Officer",
        _story(HANDBOOK_SECTIONS, HANDBOOK_TABLES),
    )
    _build(
        out_dir / "Acme_IT_Security_Policy_2026.pdf",
        "Acme Corp IT Security Policy 2026",
        "Version 4.0 — Effective 1 January 2026 — Owner: Chief Information Security Officer",
        _story(SECURITY_SECTIONS, SECURITY_TABLES),
    )
    _build(
        out_dir / "Acme_Remote_Work_Addendum_A_2026.pdf",
        "Acme Corp Remote Work Addendum A",
        "Effective 1 March 2026 — Amends Employee Handbook 2026, Section 5",
        _story(ADDENDUM_SECTIONS, {}),
    )
    print(f"Wrote 3 policy PDFs to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/policies", help="output directory")
    args = parser.parse_args()
    main(Path(args.out))
