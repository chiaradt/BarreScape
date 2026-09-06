"""
Ballet AI Technique Analysis Server
Flask backend with MediaPipe Pose (Tasks API), Gemini data filtering,
and Claude coaching critique pipeline.
"""

import os
import json
import math
import tempfile
import datetime
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_vision
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import anthropic

# ---------------------------------------------------------------------------
# Configuration — set API keys in the environment, not as placeholders
# ---------------------------------------------------------------------------
def get_env_api_key(name: str, placeholder: str) -> str:
    value = os.environ.get(name, "").strip()
    if value and value != placeholder:
        return value
    return ""


GOOGLE_API_KEY    = get_env_api_key("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
ANTHROPIC_API_KEY = get_env_api_key("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY_HERE")

PROJECT_DIR       = os.path.dirname(os.path.abspath(__file__))
HISTORY_DB_PATH   = os.path.join(PROJECT_DIR, "history_db.json")
BALLET_RULES_PATH = os.path.join(PROJECT_DIR, "ballet_rules.json")
MODEL_PATH        = os.path.join(PROJECT_DIR, "pose_landmarker_lite.task")

# MediaPipe lite model — downloaded automatically on first run
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

# Process every Nth frame for speed
FRAME_SAMPLE_RATE    = 3
STATIC_MOTION_THRESHOLD = 6.0   # avg pixel displacement — below this = static pose hold

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# ---------------------------------------------------------------------------
# Pose model bootstrap
# ---------------------------------------------------------------------------
def ensure_pose_model() -> str:
    """Download the pose landmarker model file if not already present."""
    if not os.path.exists(MODEL_PATH):
        print(f"  Downloading pose model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(POSE_MODEL_URL, MODEL_PATH)
        print("  Pose model downloaded.")
    return MODEL_PATH


# ---------------------------------------------------------------------------
# Utility: history database (local JSON)
# ---------------------------------------------------------------------------
def load_history_db() -> dict:
    if os.path.exists(HISTORY_DB_PATH):
        try:
            with open(HISTORY_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_history_db(db: dict) -> None:
    with open(HISTORY_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def get_user_history_text(db: dict, username: str) -> str:
    """Return the text of the user's most recent critique, or empty string."""
    entries = db.get(username, [])
    if entries:
        return entries[-1].get("critique", "")
    return ""


def save_user_critique(db: dict, username: str, critique: str,
                        skill_level: str, health_specs: str) -> None:
    if username not in db:
        db[username] = []
    db[username].append({
        "date":        datetime.datetime.now().isoformat(timespec="seconds"),
        "skill_level": skill_level,
        "health_specs": health_specs,
        "critique":    critique,
    })


# ---------------------------------------------------------------------------
# Utility: ballet rules (optional camera-translation lenses)
# ---------------------------------------------------------------------------
def load_ballet_rules() -> str:
    if os.path.exists(BALLET_RULES_PATH):
        try:
            with open(BALLET_RULES_PATH, "r", encoding="utf-8") as f:
                rules = json.load(f)
            return json.dumps(rules, indent=2)
        except (json.JSONDecodeError, IOError):
            pass
    return (
        "No custom camera translation rules file found (ballet_rules.json). "
        "Rely fully on your native classical ballet knowledge."
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def angle_between(a, b, c) -> float:
    """
    Interior angle (degrees) at vertex b,
    formed by rays b→a and b→c.
    """
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    dot      = bax * bcx + bay * bcy
    mag_ba   = math.hypot(bax, bay)
    mag_bc   = math.hypot(bcx, bcy)
    if mag_ba * mag_bc < 1e-9:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))))


def euclidean(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ---------------------------------------------------------------------------
# MediaPipe landmark indices (same 33-point schema in Tasks API)
# ---------------------------------------------------------------------------
IDX = {
    "nose":           0,
    "left_shoulder":  11, "right_shoulder": 12,
    "left_elbow":     13, "right_elbow":    14,
    "left_wrist":     15, "right_wrist":    16,
    "left_hip":       23, "right_hip":      24,
    "left_knee":      25, "right_knee":     26,
    "left_ankle":     27, "right_ankle":    28,
    "left_foot":      31, "right_foot":     32,
}


def extract_frame_metrics(landmarks, frame_w: int, frame_h: int) -> dict:
    """
    Convert normalized landmarks to pixel coords and compute
    all ballet metrics for a single frame.
    """
    def px(name):
        lm = landmarks[IDX[name]]
        return (lm.x * frame_w, lm.y * frame_h)

    ls  = px("left_shoulder");  rs  = px("right_shoulder")
    lh  = px("left_hip");       rh  = px("right_hip")
    lk  = px("left_knee");      rk  = px("right_knee")
    la  = px("left_ankle");     ra  = px("right_ankle")

    # Derived centers
    hip_center      = ((lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5)
    shoulder_center = ((ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5)

    # Knee angles  (hip → knee → ankle)
    l_knee_angle = angle_between(lh, lk, la)
    r_knee_angle = angle_between(rh, rk, ra)

    # Spine lean: angle of torso from vertical (virtual point 200px above hip)
    vertical_up  = (hip_center[0], hip_center[1] - 200)
    spine_angle  = angle_between(vertical_up, hip_center, shoulder_center)
    spine_signed = spine_angle if shoulder_center[0] >= hip_center[0] else -spine_angle

    # Hip asymmetry: Y-difference between hips in image space
    # (+) = left hip lower in frame, suggesting right side elevation
    hip_asymmetry = lh[1] - rh[1]

    # Shoulder width (horizontal spread — narrows during pirouette rotation)
    shoulder_width = abs(ls[0] - rs[0])

    # Ankle gap (horizontal distance — rough turnout proxy)
    ankle_gap = abs(la[0] - ra[0])

    return {
        "left_knee_angle":   round(l_knee_angle,  1),
        "right_knee_angle":  round(r_knee_angle,  1),
        "spine_angle":       round(spine_signed,  1),
        "hip_asymmetry_px":  round(hip_asymmetry, 1),
        "shoulder_width_px": round(shoulder_width,1),
        "ankle_gap_px":      round(ankle_gap,     1),
        # Raw joint points needed for motion index computation
        "_left_shoulder":  ls, "_right_shoulder": rs,
        "_left_hip":       lh, "_right_hip":      rh,
        "_left_knee":      lk, "_right_knee":     rk,
        "_left_ankle":     la, "_right_ankle":    ra,
    }


# ---------------------------------------------------------------------------
# Motion Index
# ---------------------------------------------------------------------------
def compute_motion_index(prev: dict, curr: dict) -> float:
    """
    Average Euclidean pixel displacement across 8 tracked joints
    between two consecutively sampled frames.
    """
    joints = [
        "_left_shoulder", "_right_shoulder",
        "_left_hip",      "_right_hip",
        "_left_knee",     "_right_knee",
        "_left_ankle",    "_right_ankle",
    ]
    total = sum(euclidean(prev[j], curr[j]) for j in joints)
    return round(total / len(joints), 2)


# ---------------------------------------------------------------------------
# Video processor (MediaPipe Tasks API)
# ---------------------------------------------------------------------------
def process_video(video_path: str):
    """
    Run MediaPipe PoseLandmarker over the video in VIDEO running mode.
    Returns (frames_data, fps, duration_s).
    """
    model_path = ensure_pose_model()

    base_opts = mp_tasks_python.BaseOptions(model_asset_path=model_path)
    landmarker_opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        output_segmentation_masks=False,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames_data    = []
    raw_frame_idx  = 0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    with mp_vision.PoseLandmarker.create_from_options(landmarker_opts) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            raw_frame_idx += 1
            if raw_frame_idx % FRAME_SAMPLE_RATE != 0:
                continue

            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # VIDEO mode requires monotonically increasing timestamps in ms
            timestamp_ms = int((raw_frame_idx / fps) * 1000)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                pose_lms = result.pose_landmarks[0]  # first detected person
                metrics  = extract_frame_metrics(pose_lms, w, h)
                metrics["frame_idx"] = raw_frame_idx
                metrics["time_s"]    = round(raw_frame_idx / fps, 2)
                frames_data.append(metrics)

    cap.release()
    duration = (raw_frame_idx / fps) if raw_frame_idx > 0 else 0
    return frames_data, fps, duration


# ---------------------------------------------------------------------------
# Timeline summary builder
# ---------------------------------------------------------------------------
def build_timeline_summary(frames_data: list, duration_s: float) -> str:
    if not frames_data:
        return (
            "WARNING: MediaPipe could not detect a human pose in this video.\n"
            "Ensure the dancer's full or upper body is visible, well-lit, "
            "and wearing form-fitting dancewear."
        )

    # Compute motion index between consecutive sampled frames
    motion_series = []
    for i in range(1, len(frames_data)):
        mi = compute_motion_index(frames_data[i - 1], frames_data[i])
        motion_series.append((frames_data[i]["time_s"], mi))

    # Segment into STATIC_POSE / DYNAMIC_MOVE phases
    phases = []
    if motion_series:
        current_phase = "STATIC_POSE" if motion_series[0][1] < STATIC_MOTION_THRESHOLD else "DYNAMIC_MOVE"
        phase_start   = motion_series[0][0]
        phase_times   = []

        for (t, mi) in motion_series:
            p = "STATIC_POSE" if mi < STATIC_MOTION_THRESHOLD else "DYNAMIC_MOVE"
            if p != current_phase:
                phases.append((current_phase, phase_start, t, phase_times[:]))
                current_phase = p
                phase_start   = t
                phase_times   = []
            phase_times.append(t)

        if phase_times:
            phases.append((current_phase, phase_start, duration_s, phase_times))

    # Aggregate metrics across all frames
    def agg(key):
        vals = [f[key] for f in frames_data]
        return min(vals), max(vals), round(sum(vals) / len(vals), 1)

    lk_min, lk_max, lk_avg = agg("left_knee_angle")
    rk_min, rk_max, rk_avg = agg("right_knee_angle")
    sp_min, sp_max, sp_avg = agg("spine_angle")
    ha_min, ha_max, ha_avg = agg("hip_asymmetry_px")
    sw_min, sw_max, sw_avg = agg("shoulder_width_px")
    ag_min, ag_max, ag_avg = agg("ankle_gap_px")

    lines = [
        "BALLET VIDEO POSE ANALYSIS",
        f"Sampled frames: {len(frames_data)}  |  Duration: {duration_s:.1f}s",
        f"Static motion threshold: {STATIC_MOTION_THRESHOLD}px avg displacement",
        "",
        "OVERALL METRIC RANGES (all sampled frames):",
        f"  Left  Knee Angle : min={lk_min}°  max={lk_max}°  avg={lk_avg}°",
        f"  Right Knee Angle : min={rk_min}°  max={rk_max}°  avg={rk_avg}°",
        f"  Spine Lean (±=R/L): min={sp_min}°  max={sp_max}°  avg={sp_avg}°",
        f"  Hip Asymmetry (px): min={ha_min}  max={ha_max}  avg={ha_avg}  (+ = left hip lower)",
        f"  Shoulder Width (px): min={sw_min}  max={sw_max}  avg={sw_avg}",
        f"  Ankle Spread (px): min={ag_min}  max={ag_max}  avg={ag_avg}",
        "",
        "MOVEMENT PHASE TIMELINE:",
    ]

    for (phase_type, t_start, t_end, p_times) in phases[:25]:
        seg_frames = [f for f in frames_data if f["time_s"] in p_times]
        if not seg_frames:
            seg_frames = frames_data
        s_lk = round(sum(f["left_knee_angle"]   for f in seg_frames) / len(seg_frames), 1)
        s_rk = round(sum(f["right_knee_angle"]  for f in seg_frames) / len(seg_frames), 1)
        s_sp = round(sum(f["spine_angle"]        for f in seg_frames) / len(seg_frames), 1)
        s_ha = round(sum(f["hip_asymmetry_px"]   for f in seg_frames) / len(seg_frames), 1)
        s_sw = round(sum(f["shoulder_width_px"]  for f in seg_frames) / len(seg_frames), 1)
        lines.append(
            f"  [{phase_type}] {t_start:.1f}s–{t_end:.1f}s  "
            f"L-Knee={s_lk}°  R-Knee={s_rk}°  Spine={s_sp}°  "
            f"HipTilt={s_ha}px  ShoulderW={s_sw}px"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM Pipeline
# ---------------------------------------------------------------------------
def run_gemini_filter(timeline_text: str) -> str:
    """
    Step 1 — Gemini 2.5 Flash: objective data filter that extracts
    a bulleted list of technical errors from raw tracking data.
    """
    key = GOOGLE_API_KEY or ""
    print(f"[DEBUG] GOOGLE_API_KEY used: first12={key[:12]} | last4={key[-4:]} | len={len(key)}")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    prompt = f"""You are a precise, objective motion data analysis filter for classical ballet.

Your ONLY task is to read the raw MediaPipe pose tracking timeline below and output a clean, bulleted list of observable technical irregularities and body shape deviations that the numbers reveal.

Rules:
- Be purely descriptive and data-driven. No coaching language, no advice.
- At each detected hold or transition, attempt to identify the actual ballet step or position from the body-shape and joint-angle pattern, such as 'arabesque', 'attitude', 'chassé', 'tendu', 'plié', 'relevé', 'pirouette', 'grand battement', 'passé', 'a la seconde', 'first position', 'second position', 'fifth position', or 'demi-plié'.
- For standing or basic floor positions, specifically look for classic ballet terminology when the joint/turnout pattern clearly indicates it: 'first position', 'second position', 'fifth position', 'demi-plié', 'grand plié', 'relevé in first position', or 'simple standing alignment in first position'. Do not fall back to vague biomechanical labels like 'two-leg stance' when the data clearly supports a real ballet position.
- If the exact step cannot be identified confidently, describe the general body shape instead, such as 'one leg extended behind at hip height', 'single-leg balance with knee lifted', 'straight supporting leg with torso rotated', or 'small side step with weight shifting'. Do not use vague labels like 'static pose' or 'transition' as the primary name when a more specific motion or shape is visible.
- When the same step or position appears more than once, number the instances in order as they appear in the video, such as '1st arabesque', '2nd arabesque', '1st tendu', '2nd tendu', '1st fifth position', '2nd fifth position'. Use the same numbering consistently across the list.
- Each bullet must name the error type and describe what the metric concretely shows, and should include the most likely step/position label or the general body shape description so later coaching can reference specific movements rather than generic timestamps alone.
- Group related issues logically (e.g. all knee observations together).
- Only flag clear anomalies — do NOT mention metrics within normal range.
- Maximum 12 bullets.

RAW TRACKING DATA:
{timeline_text}

Output format — return ONLY this bulleted list, nothing else:
• [ERROR TYPE]: [Step/position label or general shape] — objective description of what the metric shows
• [ERROR TYPE]: ...
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )
    return response.text.strip()


def run_claude_coaching(
    gemini_summary:    str,
    username:          str,
    skill_level:       str,
    health_specs:      str,
    history_data:      str,
    ballet_rules_text: str,
) -> str:
    """
    Step 2 — Claude 3.5 Sonnet: transforms Gemini's error list into
    a warm, structured somatic coaching critique.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    system_prompt = f"""You are an expert classical ballet coach and somatic alignment specialist trained in both the Vaganova and RAD methodologies. You are warm, clear, encouraging, and highly precise.

━━━ NATIVE BALLET KNOWLEDGE (PRIMARY BRAIN) ━━━
Rely fully on your encyclopedic, native understanding of classical ballet terminology, positions, placement, and anatomy. You know what a correct Plié, Arabesque, Attitude, Pas de Basque, Grand Battement, Pirouette, Port de Bras, and all fundamental positions look and feel like. Use this knowledge as your primary evaluative framework.

━━━ CUSTOM CAMERA TRANSLATION LENSES (EQUAL WEIGHT) ━━━
{ballet_rules_text}
Treat these as equal-weight companion tools to help you decode raw screen metrics (e.g. reading opposite shoulder depth to diagnose a twisted hip, or tracking velocity curves to identify slackened glutes). They are additive lenses — they do not suppress your native knowledge.

━━━ COACHING TONE ━━━
- Clear, encouraging, matter-of-fact, and highly constructive. Never scolding, harsh, or dramatic.
- State errors plainly and immediately follow with a practical correction.
- Write as a supportive, experienced studio teacher would speak.
- Default to authentic ballet studio cueing language used by real teachers in class. Translate any clinical, biomechanical, or fitness phrasing into how a teacher would actually say it in the room.

━━━ OUTPUT RULES ━━━
- Absolute formatting rule: Wrap only the issue title and the labels Immediate correction: and Long-term training guide: in double asterisks, like **this**. Nothing else in the response should ever be wrapped in asterisks.
- Absolute ban on numeric angle values in dancer-facing text: decimals, degree symbols, exact measured angles, and any specific numeric angle values must never appear in the text shown to the dancer. This includes values such as '155.3°', '132.6°', '90.4°', '0.2s-2.1s', or other raw angle/measurement numbers. The only exception is standard ballet extension language spoken as real ballet vocabulary, such as 45°, 90°, 180°, or phrases like 'just below 90' or 'close to 90'. All other angle information is internal reasoning only and must be translated into qualitative studio-language descriptions.
- Translate internal angle calculations into coachable language. Instead of giving raw numbers, describe what the dancer is actually doing in the room in terms of pattern, timing, and quality: 'your right knee still isn't finishing its straighten', 'the leg is dropping early in the phrase', 'the hip is collapsing as the turn begins', 'the same pattern is repeating in the same section as last time', 'the chest is drifting forward before the leg opens', 'the back leg is losing the line instead of staying long and lifted'. Describe patterns and consistency compared with history, not number-to-number comparisons.
- If you need to mention timing in a video, use natural spoken timestamps such as 'about 4 seconds in' or 'near the end of the phrase', not exact decimal ranges. Round to natural spoken language and keep the focus on where in the movement to look, not on raw numeric measurements.
- Name the actual step or movement in your coaching language, such as 'tendu', 'plié', 'arabesque', 'attitude', 'relevé', 'grand battement', 'pirouette', or 'port de bras'. Do not use generic labels like 'static pose', 'position', or 'alignment issue' when the context clearly indicates a specific movement. Infer the movement from context if it is not explicitly named.
- Use teacher-like cues, not generic fitness or anatomy phrases. Prefer real studio language such as 'pull up the kneecap', 'stand taller through the crown', 'keep the hip over the foot', 'lengthen the back of the neck', 'reach the toes', 'lift the chest without pinching the lower back', 'bring the shoulder blade down and wide', 'straighten the standing leg', 'keep the weight over the middle of the foot', 'soften the knee', and 'draw the leg out from the hip'. Avoid clinical or fitness wording such as 'engage', 'activate', 'deviation', 'compression', 'stabilize', 'anterior pelvic tilt', 'deviated alignment', 'muscle activation', or other generic exercise-language phrasing.
- When the Gemini summary names a specific step or position with numbering, use that exact movement naming in the critique and refer to it as '1st arabesque', '2nd attitude', or whichever label appears. Do not collapse the feedback back into generic timestamps alone when a numbered step label is available.
- Be honest about progress. Only call out improvement when the data shows a genuine, meaningful difference from the previous session. If the metrics are nearly identical to last time, say that plainly and do not fabricate praise.

━━━ PHYSICAL CAPABILITY COMPASSION RULE ━━━
Dancer health/physical notes: {health_specs if health_specs else 'None provided.'}
Skill level: {skill_level}

Interpret the health_specs input as two different categories:
1. Physical limitations, injuries, or structural restrictions to account for compassionately and safely.
2. Specific technique focus areas the dancer wants extra attention on, such as 'please focus on my turnout', 'I am working on my arabesque line', 'I want more port de bras', or similar direct requests.

For category 1: For beginners and intermediates — DO NOT penalize natural physical limitations such as:
- Tight ankles limiting foot arch or pointed-toe depth
- Limited hamstring flexibility restricting extension height
- Naturally reduced turnout from hip socket structure
- Restricted spinal mobility

Acknowledge these restrictions compassionately. Adjust all alignment checks to prioritise absolute physical safety first. If a restriction is structural or injury-related, never frame it as a careless mistake.

For category 2: If the dancer explicitly requests a technique focus area, give that area priority in the critique. When a focus area is mentioned, expand on relevant findings within that area, explain how it is appearing in the movement, and give practical correction cues and training guidance that directly serve that goal. If other issues are present, still address them, but make the dancer's requested focus area a clear emphasis in the feedback and in the coaching priorities.

━━━ PAST HISTORY ACKNOWLEDGEMENT ━━━
Previous session critique for this dancer:
{history_data if history_data else "This is the dancer's first session on record — no historical comparison available."}

If history exists: Mention improvement only when there is a clear, meaningful difference supported by the data. If there is no meaningful change, say so plainly rather than praising a false improvement.

━━━ OUTPUT STRUCTURE ━━━
Begin with 1-2 short, warm sentences greeting the dancer by name and setting a positive tone before the first issue is listed.

For each identified issue, use this structure:

**[Issue Name — brief, plain English]**
**Immediate correction:** A practical, muscle-engagement studio cue to safely adjust their posture right now, within their current physical capacity.
**Long-term training guide:** A specific, safe 6-week progressive exercise or conditioning stretch to address the root cause. If the issue is structural or injury-related, explicitly ask whether they would like tailored modifications, and offer 1-2 safe low-impact variants appropriate for their current capacity.

If more than four issues are detected, prioritise and discuss only the most significant ones in full detail. Briefly acknowledge any additional minor observations in one short closing sentence rather than giving each equal in-depth treatment. Focus on the issues that most affect technique, safety, and the dancer's stated focus area.

After all issues, close with a brief, warm, personalised encouragement note that references their specific strengths and any genuine progress visible in this session. If there is no meaningful change from the last session, state that plainly instead of exaggerating progress."""

    user_message = f"""Dancer name: {username}
Skill level: {skill_level}
Physical notes / health specs: {health_specs if health_specs else 'None.'}

━━━ GEMINI TECHNICAL ERROR SUMMARY ━━━
{gemini_summary}

Please produce your full ballet coaching critique following the system output structure."""

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    text_parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)

    raw_text = "".join(text_parts)
    print("[DEBUG] Claude raw response start")
    print(raw_text)
    print("[DEBUG] Claude raw response end")
    return raw_text


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.route("/upload-ballet", methods=["POST"])
def upload_ballet():
    # Extract form parameters
    username     = (request.form.get("username")     or "anonymous").strip()
    skill_level  = (request.form.get("skill_level")  or "Adult / Recreational Beginner").strip()
    health_specs = (request.form.get("health_specs") or "").strip()

    # Validate video file
    if "video" not in request.files:
        return jsonify({"error": "No video file provided. Include a 'video' field in the multipart form."}), 400

    video_file = request.files["video"]
    if not video_file or video_file.filename == "":
        return jsonify({"error": "Empty video file received."}), 400

    # Save to a temp file
    suffix   = os.path.splitext(video_file.filename or ".mp4")[1] or ".mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)

    try:
        video_file.save(tmp_path)

        # Load history and ballet rules
        history_db   = load_history_db()
        history_data = get_user_history_text(history_db, username)
        ballet_rules = load_ballet_rules()

        # Process video with MediaPipe
        frames_data, fps, duration_s = process_video(tmp_path)

        if not frames_data:
            return jsonify({
                "corrections": (
                    "No pose was detected in your video. Please ensure:\n"
                    "• The dancer's body (hips to head) is clearly visible throughout.\n"
                    "• The room is well-lit with no strong backlighting.\n"
                    "• You are wearing form-fitting dancewear (leotard and tights).\n\n"
                    "Please re-upload with these conditions corrected."
                )
            })

        timeline_text = build_timeline_summary(frames_data, duration_s)

        # Step 1: Gemini data filter
        gemini_summary = run_gemini_filter(timeline_text)

        # Step 2: Claude coaching critique
        critique = run_claude_coaching(
            gemini_summary    = gemini_summary,
            username          = username,
            skill_level       = skill_level,
            health_specs      = health_specs,
            history_data      = history_data,
            ballet_rules_text = ballet_rules,
        )

        # Persist to history
        save_user_critique(history_db, username, critique, skill_level, health_specs)
        save_history_db(history_db)

        return jsonify({"corrections": critique})

    except Exception as exc:
        return jsonify({"error": f"Server error: {str(exc)}"}), 500

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "Ballet AI server is running.", "version": "1.0.0"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  Ballet AI Technique Analysis Server  v1.0")
    print("=" * 62)
    print(f"  Google API Key   : {'SET' if GOOGLE_API_KEY   != 'YOUR_GOOGLE_API_KEY_HERE'   else 'NOT SET — add to GOOGLE_API_KEY env var'}")
    print(f"  Anthropic Key    : {'SET' if ANTHROPIC_API_KEY != 'YOUR_ANTHROPIC_API_KEY_HERE' else 'NOT SET — add to ANTHROPIC_API_KEY env var'}")
    print(f"  History DB       : {HISTORY_DB_PATH}")
    print(f"  Ballet Rules     : {BALLET_RULES_PATH} ({'found' if os.path.exists(BALLET_RULES_PATH) else 'not found — optional'})")
    print(f"  Pose Model       : {MODEL_PATH} ({'ready' if os.path.exists(MODEL_PATH) else 'will download on first request'})")
    print("=" * 62)
    print("  Server           : http://127.0.0.1:5000")
    print("  Upload endpoint  : POST http://127.0.0.1:5000/upload-ballet")
    print("  Health check     : GET  http://127.0.0.1:5000/ping")
    print("=" * 62 + "\n")

    app.run(debug=True, host="127.0.0.1", port=5000)
