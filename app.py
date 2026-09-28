from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.database import Database
from src.imaging import assess_quality, image_fingerprint
from src.inference import get_grader
from src.report import build_report
from src.triage import determine_triage

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "data" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="NetraSetu", page_icon="👁️", layout="wide")


@st.cache_resource
def resources():
    return Database(ROOT / "data" / "app.db"), get_grader(str(ROOT))


db, grader = resources()

st.markdown(
    """
    <style>
    .stApp { background: #f7faf8; }
    [data-testid="stSidebar"] { background: #102c25; }
    [data-testid="stSidebar"] * { color: #f7fbf9; }
    .hero { padding: 1.4rem 1.6rem; border-radius: 18px; color: white;
            background: linear-gradient(120deg,#123b31,#176a55); margin-bottom: 1rem; }
    .hero h1 { margin: 0; font-size: 2rem; font-weight: 650; }
    .hero p { margin: .35rem 0 0; color: #d9eee7; }
    .eyebrow { color:#3d6a5d; font-size:.76rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    .priority { display:inline-block; padding:.28rem .7rem; border-radius:999px; font-weight:700; font-size:.82rem; }
    .urgent { color:#8d1c1c; background:#fee2e2; }
    .priority-level { color:#854d0e; background:#fef3c7; }
    .referral { color:#1e40af; background:#dbeafe; }
    .routine { color:#166534; background:#dcfce7; }
    .manual { color:#5b21b6; background:#ede9fe; }
    .prototype { padding:.65rem .85rem; border-left:4px solid #d97706; background:#fff7ed; border-radius:6px; color:#7c2d12; }
    div[data-testid="stMetric"] { background:white; border:1px solid #dfe9e5; padding:.8rem; border-radius:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 👁️ NetraSetu")
    st.caption("Rural diabetic-retinopathy screening")
    role = st.radio("Open workspace", ["Village health centre", "Ophthalmologist"], label_visibility="collapsed")
    st.divider()
    model_text = "Trained checkpoint" if grader.model is not None else "Transparent APTOS demo mode"
    st.markdown(f"**Screening engine**  \n{model_text}")
    st.caption("Research prototype • Not an autonomous diagnosis")


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def priority_badge(priority: str) -> str:
    css = {
        "URGENT": "urgent",
        "PRIORITY": "priority-level",
        "REFERRAL": "referral",
        "ROUTINE": "routine",
        "MANUAL REVIEW": "manual",
    }.get(priority, "manual")
    return f'<span class="priority {css}">{priority}</span>'


def save_upload(upload, eye: str) -> Path:
    suffix = Path(upload.name).suffix.lower() or ".png"
    path = UPLOADS / f"{uuid.uuid4().hex}_{eye}{suffix}"
    path.write_bytes(upload.getvalue())
    return path


def empty_result(message: str) -> dict:
    return {
        "grade": None,
        "label": message,
        "probabilities": [],
        "confidence": 0.0,
        "low_confidence": True,
        "mode": "quality-gate",
        "model_version": "not-run",
        "warning": message,
    }


def show_quality(label: str, quality: dict) -> None:
    if quality["accepted"]:
        st.success(f"{label}: image accepted")
    else:
        st.error(f"{label}: recapture recommended")
        for problem in quality["problems"]:
            st.write(f"• {problem}")
        for guidance in quality["guidance"]:
            st.caption(guidance)
    metrics = quality["metrics"]
    st.caption(
        f'Blur {metrics["blur_score"]} · Brightness {metrics["brightness"]} · '
        f'Field coverage {metrics["field_coverage"]:.0%}'
    )


def seed_demonstration_cases() -> int:
    if db.list_patients():
        return 0
    label_csv = ROOT / "data" / "raw" / "aptos" / "train.csv"
    image_dir = ROOT / "data" / "raw" / "aptos" / "train_images"
    if not label_csv.exists() or not image_dir.exists():
        return 0
    labels = pd.read_csv(label_csv)
    selected: dict[int, list[Path]] = {}
    for grade in (0, 2, 4):
        accepted: list[Path] = []
        for image_id in labels.loc[labels["diagnosis"] == grade, "id_code"].head(80):
            path = image_dir / f"{image_id}.png"
            if path.exists() and assess_quality(path)["accepted"]:
                accepted.append(path)
            if len(accepted) == 2:
                break
        if len(accepted) < 2:
            accepted = [image_dir / f"{value}.png" for value in labels.loc[labels["diagnosis"] == grade, "id_code"].head(2)]
        selected[grade] = accepted

    people = [
        ("NS-1001", "Kamala Devi", 58, "Female", "Rampur", 0, []),
        ("NS-1002", "Ramesh Kumar", 63, "Male", "Lakshmipur", 2, ["Blurred vision"]),
        ("NS-1003", "Savitri Bai", 67, "Female", "Rampur", 4, ["New flashes or many floaters"]),
    ]
    for index, (local_id, name, age, sex, village, grade, symptoms) in enumerate(people):
        patient_id = db.create_patient(
            {
                "local_patient_id": local_id,
                "name": name,
                "age": age,
                "sex": sex,
                "phone": f"90000000{index + 1:02d}",
                "village": village,
                "diabetes_duration": 4 + index * 4,
                "consent": True,
            }
        )
        right_path, left_path = selected[grade]
        right_quality, left_quality = assess_quality(right_path), assess_quality(left_path)
        right_result, left_result = grader.predict(right_path), grader.predict(left_path)
        triage = determine_triage(
            right_result["grade"], left_result["grade"], symptoms,
            low_confidence=right_result["low_confidence"] or left_result["low_confidence"],
        )
        db.create_screening(
            {
                "patient_id": patient_id,
                "right_image": str(right_path),
                "left_image": str(left_path),
                "right_quality": right_quality,
                "left_quality": left_quality,
                "right_result": right_result,
                "left_result": left_result,
                "history": {
                    "blood_pressure": ["126/78", "148/92", "156/94"][index],
                    "visual_acuity_right": ["6/6", "6/12", "6/24"][index],
                    "visual_acuity_left": ["6/6", "6/9", "6/18"][index],
                    "symptoms": symptoms,
                    "notes": "Synthetic hackathon demonstration case",
                },
                "triage": triage,
            }
        )
    return len(people)


def village_dashboard() -> None:
    hero("Village health-centre workspace", "Register, capture, screen and close the referral loop.")
    tabs = st.tabs(["Overview", "New screening", "Patients & follow-up"])
    with tabs[0]:
        counts = db.counts()
        cols = st.columns(5)
        for column, label, value in zip(
            cols,
            ["Registered patients", "Awaiting doctor", "Urgent action", "Reviewed", "Recapture"],
            [counts["patients"], counts["awaiting"], counts["urgent"], counts["reviewed"], counts["recapture"]],
        ):
            column.metric(label, value)
        st.subheader("Today’s action list")
        cases = db.list_screenings()
        actionable = [case for case in cases if case["status"] != "Doctor reviewed"]
        if actionable:
            frame = pd.DataFrame(
                [
                    {
                        "Patient": case["patient_name"],
                        "Local ID": case["local_patient_id"],
                        "Priority": case["priority"],
                        "Status": case["status"],
                        "Village": case["village"],
                    }
                    for case in actionable
                ]
            )
            st.dataframe(frame, width="stretch", hide_index=True)
        else:
            st.info("No active cases yet. Start a screening or load the demonstration cases.")
        if not db.list_patients():
            if st.button("Load three demonstration cases", type="primary"):
                count = seed_demonstration_cases()
                st.success(f"Loaded {count} synthetic demonstration cases.")
                st.rerun()

    with tabs[1]:
        render_new_screening()

    with tabs[2]:
        render_patient_followup()


def render_new_screening() -> None:
    st.markdown('<div class="eyebrow">Guided capture</div>', unsafe_allow_html=True)
    st.subheader("New retinal screening")
    patients = db.list_patients()
    choices = {f'{patient["local_patient_id"]} · {patient["name"]}': patient for patient in patients}
    patient_choice = st.selectbox("Patient", ["Create a new patient"] + list(choices))
    creating = patient_choice == "Create a new patient"

    with st.form("screening_form", clear_on_submit=False):
        if creating:
            first, second = st.columns(2)
            local_id = first.text_input("Local patient ID", value=f"NS-{len(patients) + 1001}")
            name = second.text_input("Patient name")
            age = first.number_input("Age", min_value=18, max_value=110, value=55)
            sex = second.selectbox("Sex", ["Female", "Male", "Other"])
            phone = first.text_input("Phone number")
            village = second.text_input("Village")
            abha = first.text_input("ABHA number (optional)")
            diabetes_duration = second.number_input("Years with diabetes", min_value=0, max_value=80, value=5)
            consent = st.checkbox("Patient consent recorded", value=False)
        else:
            patient = choices[patient_choice]
            st.info(f'{patient["name"]}, {patient["age"]} years · {patient["village"] or "Village not recorded"}')
            consent = bool(patient["consent"])

        st.markdown("#### Clinical information")
        col1, col2, col3 = st.columns(3)
        blood_pressure = col1.text_input("Blood pressure", placeholder="e.g. 130/80")
        visual_right = col2.text_input("Right-eye visual acuity", placeholder="e.g. 6/9")
        visual_left = col3.text_input("Left-eye visual acuity", placeholder="e.g. 6/9")
        symptoms = st.multiselect(
            "Symptoms",
            ["Blurred vision", "Sudden vision loss", "Curtain-like shadow", "New flashes or many floaters", "Eye pain", "No symptoms"],
        )
        notes = st.text_area("Additional history", placeholder="Previous eye treatment, pregnancy, current medicines…")

        st.markdown("#### Fundus images")
        right_col, left_col = st.columns(2)
        right_upload = right_col.file_uploader("Right eye", type=["png", "jpg", "jpeg"])
        left_upload = left_col.file_uploader("Left eye", type=["png", "jpg", "jpeg"])
        repeated_failure = st.checkbox("Multiple capture attempts have already failed; allow ungradable referral")
        submitted = st.form_submit_button("Submit screening", type="primary", width="stretch")

    if not submitted:
        return
    if not consent:
        st.error("Record patient consent before submitting.")
        return
    if not right_upload or not left_upload:
        st.error("Upload both right- and left-eye images.")
        return
    if creating and (not name.strip() or not local_id.strip()):
        st.error("Patient name and local patient ID are required.")
        return

    try:
        if creating:
            patient_id = db.create_patient(
                {
                    "local_patient_id": local_id.strip(), "name": name.strip(), "age": int(age), "sex": sex,
                    "phone": phone.strip(), "village": village.strip(), "abha_number": abha.strip(),
                    "diabetes_duration": int(diabetes_duration), "consent": True,
                }
            )
        else:
            patient_id = patient["id"]

        right_path, left_path = save_upload(right_upload, "R"), save_upload(left_upload, "L")
        right_quality, left_quality = assess_quality(right_path), assess_quality(left_path)
        if image_fingerprint(right_path) == image_fingerprint(left_path):
            left_quality["accepted"] = False
            left_quality["problems"].append("Same image appears to have been uploaded for both eyes")
            left_quality["guidance"].append("Capture and upload the left eye separately.")

        quality_passed = right_quality["accepted"] and left_quality["accepted"]
        if quality_passed:
            right_result, left_result = grader.predict(right_path), grader.predict(left_path)
        else:
            right_result = empty_result("Image-quality gate did not pass")
            left_result = empty_result("Image-quality gate did not pass")

        triage = determine_triage(
            right_result["grade"], left_result["grade"], symptoms,
            ungradable=repeated_failure or not quality_passed,
            low_confidence=right_result["low_confidence"] or left_result["low_confidence"],
        )
        status = "Awaiting doctor review" if quality_passed or repeated_failure else "Recapture required"
        screening_id = db.create_screening(
            {
                "patient_id": patient_id, "right_image": str(right_path), "left_image": str(left_path),
                "right_quality": right_quality, "left_quality": left_quality,
                "right_result": right_result, "left_result": left_result,
                "history": {
                    "blood_pressure": blood_pressure, "visual_acuity_right": visual_right,
                    "visual_acuity_left": visual_left, "symptoms": symptoms, "notes": notes,
                },
                "triage": triage, "status": status,
            }
        )
        st.success(f"Screening saved · Case {screening_id[:8]}")
        q1, q2 = st.columns(2)
        with q1:
            show_quality("Right eye", right_quality)
        with q2:
            show_quality("Left eye", left_quality)
        st.markdown(priority_badge(triage["priority"]), unsafe_allow_html=True)
        for reason in triage["reasons"]:
            st.write(f"• {reason}")
        if quality_passed:
            result_cols = st.columns(2)
            result_cols[0].metric("Right-eye preliminary grade", right_result["grade"] if right_result["grade"] is not None else "Review")
            result_cols[1].metric("Left-eye preliminary grade", left_result["grade"] if left_result["grade"] is not None else "Review")
            warning = right_result.get("warning") or left_result.get("warning")
            if warning:
                st.warning(warning)
    except Exception as error:
        st.error(f"Could not save the screening: {error}")


def render_patient_followup() -> None:
    cases = db.list_screenings()
    if not cases:
        st.info("No screening records yet.")
        return
    patient_options = {f'{case["local_patient_id"]} · {case["patient_name"]} · {case["created_at"][:10]}': case for case in cases}
    selected = st.selectbox("Select screening", list(patient_options), key="followup_case")
    case = patient_options[selected]
    st.markdown(priority_badge(case["priority"]), unsafe_allow_html=True)
    st.write(f'**Status:** {case["status"]}')
    if case["status"] == "Doctor reviewed":
        st.success(f'Doctor action: {case["review_action"]}')
        st.write(case["review_notes"] or "No additional clinical notes.")
        if case["follow_up_date"]:
            st.write(f'**Follow-up:** {case["follow_up_date"]}')
        pdf = build_report(case)
        st.download_button(
            "Download doctor-reviewed report", pdf,
            file_name=f'NetraSetu_{case["local_patient_id"]}.pdf', mime="application/pdf", type="primary",
        )
    elif case["status"] == "Recapture requested":
        st.warning("The doctor requested new retinal images.")
    else:
        st.info("The case is waiting for ophthalmologist review.")


def doctor_dashboard() -> None:
    hero("Ophthalmologist workspace", "Review original images, verify the grade and return an actionable plan.")
    cases = db.list_screenings()
    counts = db.counts()
    cols = st.columns(4)
    cols[0].metric("Urgent", counts["urgent"])
    cols[1].metric("Awaiting review", counts["awaiting"])
    cols[2].metric("Recapture tasks", counts["recapture"])
    cols[3].metric("Reviewed", counts["reviewed"])
    if not cases:
        st.info("No cases are available. Load demonstration cases from the village dashboard.")
        return

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Review queue")
        status_filter = st.selectbox("Status", ["Active", "All", "Reviewed"])
        filtered = cases
        if status_filter == "Active":
            filtered = [case for case in cases if case["status"] != "Doctor reviewed"]
        elif status_filter == "Reviewed":
            filtered = [case for case in cases if case["status"] == "Doctor reviewed"]
        if not filtered:
            st.info("No cases match this filter.")
            return
        options = {
            f'{case["priority"]} · {case["patient_name"]} · {case["local_patient_id"]}': case
            for case in filtered
        }
        selected = st.radio("Select case", list(options), label_visibility="collapsed")
        case = options[selected]
    with right:
        render_case_review(case)


def render_case_review(case: dict) -> None:
    st.markdown(priority_badge(case["priority"]), unsafe_allow_html=True)
    st.subheader(f'{case["patient_name"]} · {case["local_patient_id"]}')
    st.caption(f'{case["age"]} years · {case["sex"]} · {case["village"] or "Village not recorded"}')
    image_cols = st.columns(2)
    for column, label, path_key, result_key, quality_key in (
        (image_cols[0], "Right eye", "right_image", "right_result", "right_quality"),
        (image_cols[1], "Left eye", "left_image", "left_result", "left_quality"),
    ):
        with column:
            st.markdown(f"**{label}**")
            if case.get(path_key) and Path(case[path_key]).exists():
                st.image(case[path_key], width="stretch")
            result = case[result_key]
            grade_text = result["grade"] if result["grade"] is not None else "Not graded"
            st.metric("Preliminary grade", grade_text, result["label"])
            show_quality(label, case[quality_key])

    history = case["history"]
    with st.expander("Clinical information", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.write(f'**Blood pressure**  \n{history.get("blood_pressure") or "Not recorded"}')
        c2.write(f'**Right visual acuity**  \n{history.get("visual_acuity_right") or "Not recorded"}')
        c3.write(f'**Left visual acuity**  \n{history.get("visual_acuity_left") or "Not recorded"}')
        st.write("**Symptoms:** " + (", ".join(history.get("symptoms", [])) or "None recorded"))
        if history.get("notes"):
            st.write("**Additional history:** " + history["notes"])
        st.write("**Triage reason:** " + "; ".join(case["triage_reasons"]))

    if case["right_result"].get("warning") or case["left_result"].get("warning"):
        st.markdown(
            '<div class="prototype">This case uses the APTOS reference label for the hackathon demonstration. '
            'It must not be represented as a trained model prediction.</div>',
            unsafe_allow_html=True,
        )

    current_right = case["right_result"].get("grade")
    current_left = case["left_result"].get("grade")
    grade_options = ["Ungradable", 0, 1, 2, 3, 4]
    with st.form(f'review_{case["id"]}'):
        st.markdown("#### Clinical decision")
        rcol, lcol = st.columns(2)
        right_default = grade_options.index(current_right) if current_right in grade_options else 0
        left_default = grade_options.index(current_left) if current_left in grade_options else 0
        confirmed_right = rcol.selectbox("Confirmed right-eye grade", grade_options, index=right_default)
        confirmed_left = lcol.selectbox("Confirmed left-eye grade", grade_options, index=left_default)
        action = st.selectbox(
            "Action",
            ["Confirm screening result", "Correct screening result", "Request image recapture", "Request in-person examination", "Refer to retina specialist"],
        )
        notes = st.text_area("Clinical notes")
        prescription = st.text_area("Prescription or patient instructions")
        follow_up = st.date_input("Follow-up date", value=date.today())
        doctor_col, reg_col = st.columns(2)
        doctor_name = doctor_col.text_input("Doctor name", value="Dr. Ananya Rao")
        doctor_registration = reg_col.text_input("Registration number", value="DEMO-RMP-001")
        submit_review = st.form_submit_button("Sign and return decision", type="primary", width="stretch")
    if submit_review:
        if not doctor_name.strip() or not doctor_registration.strip():
            st.error("Doctor name and registration number are required.")
            return
        db.save_review(
            case["id"],
            {
                "doctor_name": doctor_name.strip(), "doctor_registration": doctor_registration.strip(),
                "right_grade": None if confirmed_right == "Ungradable" else int(confirmed_right),
                "left_grade": None if confirmed_left == "Ungradable" else int(confirmed_left),
                "action": action, "notes": notes, "prescription": prescription,
                "follow_up_date": follow_up.isoformat(),
            },
        )
        st.success("Doctor decision saved and returned to the village centre.")
        st.rerun()

    if case["status"] == "Doctor reviewed":
        st.download_button(
            "Download signed PDF report", build_report(case),
            file_name=f'NetraSetu_{case["local_patient_id"]}.pdf', mime="application/pdf",
        )


if role == "Village health centre":
    village_dashboard()
else:
    doctor_dashboard()
