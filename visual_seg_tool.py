"""
ROS2 Recording + Visualisation and Segmentation Tool

OVERVIEW
This Streamlit app allows you to:
1. Record ROS2 topics into rosbags
2. Convert rosbags into CSV files
3. Upload demonstrations (CSV)
4. Automatically segment actions into phases:
   - approach
   - grasp
   - move
   - release
   - retreat
5. Manually edit segmentation
6. Compare AUTO vs MANUAL segmentation
7. Export results

-----------------------------------------------------------
HOW TO RUN
    streamlit run visual_seg_tool.py
-----------------------------------------------------------
"""

import streamlit as st
import subprocess
import signal
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# STREAMLIT CONFIGURATION

st.set_page_config(layout="wide")  # Use full screen width
st.title("ROS2 Recording + LfD Segmentation Tool")

# SESSION STATE

# Stores ROS2 recording process
if "process" not in st.session_state:
    st.session_state.process = None

# Tracks whether recording is active
if "recording" not in st.session_state:
    st.session_state.recording = False

# Stores segmentation (auto or manually edited)
if "segments" not in st.session_state:
    st.session_state.segments = None

# Keeps track of last uploaded file to reset segmentation
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

# ROS2 RECORDING SECTION

st.header("ROS2 Recording")

# User input: rosbag name
bag_name = st.text_input("Recording name")

# ROS2 topics to record
TOPICS = [
    "/hand_R/applied_forces",
    "/hand_R/finger_joint_angles",
    "/hand_R/hand_pose"
]

# Build ROS2 command
cmd = ["ros2", "bag", "record", "-o", bag_name] + TOPICS

# Layout for start/stop buttons
col1, col2 = st.columns(2)

# ▶️ START RECORDING
with col1:
    if st.button("▶️ Start recording"):
        if not bag_name:
            st.error("Enter recording name!")
        elif st.session_state.recording:
            st.warning("Already recording")
        else:
            # Start ROS2 recording as subprocess
            st.session_state.process = subprocess.Popen(cmd)
            st.session_state.recording = True
            st.success(f"Recording started: {bag_name}")

# ⏹️ STOP RECORDING
with col2:
    if st.button("⏹️ Stop recording"):
        if st.session_state.process and st.session_state.recording:
            # Gracefully stop ROS2 process
            st.session_state.process.send_signal(signal.SIGINT)
            st.session_state.process.wait()
            st.session_state.recording = False
            st.success("Recording stopped")
        else:
            st.warning("No active recording")

# Recording status indicator
st.info("🔴 Recording..." if st.session_state.recording else "⚪ Idle")

# CSV CONVERSION SECTION

st.header("Tools")

if st.button("📄 Convert all rosbags to CSV"):
    try:
        with st.spinner("Converting..."):
            result = subprocess.run(
                ["python3", "convert_all_rosbags_to_csv.py"],
                capture_output=True,
                text=True
            )

        if result.returncode == 0:
            st.success("Conversion completed!")
            if result.stdout:
                st.text(result.stdout)
        else:
            st.error("Conversion failed")
            st.text(result.stderr)

    except Exception as e:
        st.error(f"Error: {e}")


# FILE UPLOAD

st.header("Segmentation")

uploaded_file = st.file_uploader("Upload CSV demonstration", type=["csv"])

# Reset segmentation if new file is uploaded
if uploaded_file is not None:
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.segments = None
        st.session_state.last_uploaded_filename = uploaded_file.name


# HELPER FUNCTIONS

def standard_layout(fig, title, min_t, max_t):
    """Apply consistent layout to all plots"""
    fig.update_layout(
        title=title,
        xaxis=dict(range=[min_t, max_t]),
        margin=dict(l=120, r=40, t=40, b=40),
        yaxis=dict(automargin=False, fixedrange=True)
    )
    return fig


def add_phase_background(fig, df, segments, phase_colors):
    """Add colored background regions for each phase"""
    for seg in segments:
        fig.add_vrect(
            x0=df["timestamp"].iloc[seg["start"]],
            x1=df["timestamp"].iloc[seg["end"]],
            fillcolor=phase_colors[seg["phase"]],
            opacity=0.1,
            line_width=0
        )
    return fig


# MAIN SEGMENTATION PIPELINE


if uploaded_file:

    # Load CSV data
    df = pd.read_csv(uploaded_file)

    # FEATURE ENGINEERING

    # Total force across all fingers + palm
    df["total_force"] = (
        df["thumb_force"] + df["index_force"] +
        df["middle_force"] + df["ring_force"] +
        df["pinky_force"] + df["palm_force"]
    )

    max_force = df["total_force"].max()

    # Automatic threshold = 10% of max force
    auto_force_threshold = 0.10 * max_force

    # Force change
    df["force_diff"] = df["total_force"].diff().fillna(0)

    # Compute velocity from position differences
    dx = df["X_pos"].diff()
    dy = df["Y_pos"].diff()
    dz = df["Z_pos"].diff()
    dt = df["timestamp"].diff() / 1000.0
    dt = dt.replace(0, np.nan)

    df["speed"] = np.sqrt(dx**2 + dy**2 + dz**2) / dt
    df["speed"] = df["speed"].fillna(0)

    # Remove noise at beginning
    df.loc[:9, "speed"] = 0

    # Finger closure metric (average stretch)
    df["finger_closure"] = df[
        ["index_mcp_stretch","middle_mcp_stretch",
         "ring_mcp_stretch","pinky_mcp_stretch"]
    ].mean(axis=1)

    closure_baseline = df["finger_closure"].iloc[:20].median()

    # Smooth main signals
    for col in ["total_force", "speed", "finger_closure"]:
        df[col] = df[col].rolling(9, center=True).mean()

    df = df.bfill().ffill()

    min_t = df["timestamp"].min()
    max_t = df["timestamp"].max()

    # SIDEBAR PARAMETERS (if needed you can manipulate parameters here)

    st.sidebar.header("Segmentation parameters")

    use_auto_force = st.sidebar.checkbox("Auto force threshold (10% max)",
                                         True)

    if use_auto_force:
        force_threshold = auto_force_threshold
        st.sidebar.write(f"Force threshold: {force_threshold:.2f}")
    else:
        force_threshold = st.sidebar.slider("Force threshold", 0.0, 200.0,
                                            30.0)

    force_drop_percent = st.sidebar.slider("Force drop threshold %",
                                           0.0, 100.0, 40.0)
    force_drop_threshold = (force_drop_percent/100) * max_force

    speed_threshold = st.sidebar.slider("Speed threshold", 0.0, 0.4, 0.1)

    closure_delta = st.sidebar.slider("Closure increase %", 0.0, 100.0, 20.0)
    closure_threshold = closure_baseline * (1 + closure_delta/100)

    min_phase_len = st.sidebar.slider("Min phase length", 1, 50, 5)
    median_window = st.sidebar.slider("Median window", 1, 30, 10)

    if st.sidebar.button("Reset manual segments"):
        st.session_state.segments = None

    # AUTOMATIC SEGMENTATION

    phases = []
    phase_counter = 0
    release_active = False
    current_phase = "approach"

    for i in range(len(df)):

        # Rolling median window
        start_idx = max(0, i - median_window + 1)

        m_closure = df["finger_closure"].iloc[start_idx:i+1].median()
        m_speed = df["speed"].iloc[start_idx:i+1].median()
        m_force = df["total_force"].iloc[start_idx:i+1].median()

        # Phase transitions (rule-based)
        if phase_counter >= min_phase_len:

            if current_phase == "approach" and m_closure >= closure_threshold:
                current_phase = "grasp"
                phase_counter = 0

            elif current_phase == "grasp":
                if m_speed > speed_threshold and m_force > force_threshold:
                    current_phase = "move"
                    phase_counter = 0

            elif current_phase in ["grasp", "move"]:
                force_dropped = m_force < (max_force - force_drop_threshold)
                slow_movement = m_speed < speed_threshold

                if force_dropped and slow_movement:
                    current_phase = "release"
                    release_active = True
                    phase_counter = 0

            elif current_phase == "release" and release_active:
                if m_force < force_threshold and m_speed > speed_threshold:
                    current_phase = "retreat"
                    release_active = False
                    phase_counter = 0

        phases.append(current_phase)
        phase_counter += 1

    df["auto_phase"] = phases

    # SEGMENT BUILDING

    auto_segments = []
    current = df["auto_phase"].iloc[0]
    start = 0

    for i in range(1, len(df)):
        if df["auto_phase"].iloc[i] != current:
            auto_segments.append({"start": start, "end": i-1,
                                  "phase": current})
            start = i
            current = df["auto_phase"].iloc[i]

    auto_segments.append({"start": start, "end": len(df)-1, "phase": current})

    # Initialize manual segments
    if st.session_state.segments is None:
        st.session_state.segments = auto_segments.copy()

    # Apply manual labels
    df["phase"] = "approach"
    for seg in st.session_state.segments:
        df.loc[seg["start"]:seg["end"], "phase"] = seg["phase"]

    # VISUALIZATION

    phase_colors = {
        "approach": "blue",
        "grasp": "green",
        "move": "gold",
        "release": "red",
        "retreat": "purple"
    }

    # Plot signals
    for name, col in [
        ("Total Force","total_force"),
        ("Speed","speed"),
        ("Finger Closure","finger_closure")
    ]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[col]))
        fig = add_phase_background(fig, df, st.session_state.segments,
                                   phase_colors)
        fig = standard_layout(fig, name, min_t, max_t)
        st.plotly_chart(fig, use_container_width=True)

    # AUTO SEGMENTATION TIMELINE

    fig_auto = go.Figure()

    for seg in auto_segments:
        fig_auto.add_trace(go.Bar(
            x=[df["timestamp"].iloc[seg["end"]] - df["timestamp"].
            iloc[seg["start"]]],
            y=[seg["phase"] + "   "],
            base=df["timestamp"].iloc[seg["start"]],
            orientation='h',
            marker=dict(color=phase_colors[seg["phase"]]),
            showlegend=False
        ))

    fig_auto = standard_layout(fig_auto, "Auto Phase Segmentation",
                               min_t, max_t)
    st.plotly_chart(fig_auto, use_container_width=True)

    # MANUAL SEGMENTATION TIMELINE

    fig_manual = go.Figure()

    for seg in st.session_state.segments:
        fig_manual.add_trace(go.Bar(
            x=[df["timestamp"].iloc[seg["end"]] - df["timestamp"].
            iloc[seg["start"]]],
            y=[seg["phase"] + "   "],
            base=df["timestamp"].iloc[seg["start"]],
            orientation='h',
            marker=dict(color=phase_colors[seg["phase"]]),
            showlegend=False
        ))

    fig_manual = standard_layout(fig_manual, "Manual Phase Segmentation",
                                 min_t, max_t)
    st.plotly_chart(fig_manual, use_container_width=True)

    # MANUAL SEGMENT EDITOR

    st.subheader("Manual Timeline Editor")

    new_segments = []

    for i, seg in enumerate(st.session_state.segments):
        c1, c2, c3 = st.columns(3)

        # User edits timestamps + phase
        s = c1.number_input(f"Start {i}",
                            value=int(df["timestamp"].iloc[seg["start"]]),
                            key=f"s{i}")
        e = c2.number_input(f"End {i}",
                            value=int(df["timestamp"].iloc[seg["end"]]),
                            key=f"e{i}")
        p = c3.selectbox(f"Phase {i}", list(phase_colors.keys()),
                         index=list(phase_colors.keys()).index(seg["phase"]),
                         key=f"p{i}")

        # Convert timestamps back to indices
        si = (df["timestamp"] - s).abs().idxmin()
        ei = (df["timestamp"] - e).abs().idxmin()

        new_segments.append({"start": si, "end": ei, "phase": p})

    # Ensure continuity
    for i in range(len(new_segments)-1):
        new_segments[i]["end"] = new_segments[i+1]["start"] - 1

    if st.button("Apply edits"):
        st.session_state.segments = new_segments
        st.success("Updated")

    # COMPARISON AUTO vs MANUAL
    st.subheader("Segmentation Comparison")

    if st.button("Compare"):

        # Phase count comparison
        # Total number of segments
        auto_phase_count = len(auto_segments)
        manual_phase_count = len(st.session_state.segments)

        # Difference between segment counts
        phase_count_diff = auto_phase_count - manual_phase_count

        st.subheader("Phase Count Comparison")

        col1, col2, col3 = st.columns(3)
        col1.metric("AUTO phases", auto_phase_count)
        col2.metric("MANUAL phases", manual_phase_count)
        col3.metric("Difference", phase_count_diff)

        #  Interpretation of results
        if phase_count_diff == 0:
            st.success("Phase counts match ✅")
        elif phase_count_diff > 0:
            st.warning("AUTO has extra phases")
        else:
            st.warning("AUTO is missing phases")

       # Boundary error (start times)
        st.subheader("Boundary Error (Phase Start Times)")

        def get_phase_starts(segments, df):
            """
            Returns list of dictionaries:
            {
                "phase": phase_name,
                "start_time": timestamp
            }
            """
            starts = []

            for seg in segments:
                starts.append({
                    "phase": seg["phase"],
                    "start_time": df["timestamp"].iloc[seg["start"]]
                })
            return starts

        # Extract start times
        auto_starts = get_phase_starts(auto_segments, df)
        manual_starts = get_phase_starts(st.session_state.segments, df)

        # Ignore "approach" because it always starts correctly
        auto_starts = [s for s in auto_starts if s["phase"] != "approach"]
        manual_starts = [s for s in manual_starts if s["phase"] != "approach"]

        min_len = min(len(auto_starts), len(manual_starts))

        phase_errors = []

        for i in range(min_len):
            auto_phase = auto_starts[i]["phase"]
            manual_phase = manual_starts[i]["phase"]

            auto_time = auto_starts[i]["start_time"]
            manual_time = manual_starts[i]["start_time"]

            error_ms = abs(auto_time - manual_time)

            phase_errors.append({
                "Phase": manual_phase,
                "AUTO start": auto_time,
                "MANUAL start": manual_time,
                "Error (ms)": error_ms
            })

        # Create dataframe
        error_df = pd.DataFrame(phase_errors)

        if not error_df.empty:

            # Mean error
            mean_error = error_df["Error (ms)"].mean()

            st.metric("Average Start Error (ms)", f"{mean_error:.1f}")

            st.subheader("Phase Start Errors")

            # Show table
            st.dataframe(error_df, use_container_width=True)

            # Visualize errors
            fig_error = go.Figure()

            fig_error.add_trace(go.Bar(
                x=error_df["Error (ms)"],
                y=error_df["Phase"],
                orientation='h',
                name="Start Error"
            ))

            fig_error.update_layout(
                title="Phase Start Time Errors (ms)",
                xaxis_title="Error (ms)",
                yaxis_title="Phase"
            )

            st.plotly_chart(fig_error, use_container_width=True)

        else:
            st.info("Not enough segments to compute start-time error")

        # Sample-by-sample agreement
        agreement = (df["auto_phase"] == df["phase"])
        accuracy = agreement.mean() * 100

        # Per-phase statistics
        comparison_stats = []

        for phase_name in ["approach", "grasp", "move", "release", "retreat"]:

            # Count samples for each phase
            auto_count = (df["auto_phase"] == phase_name).sum()
            manual_count = (df["phase"] == phase_name).sum()
            matching_count = ((df["auto_phase"] == phase_name) &
                                (df["phase"] == phase_name)).sum()

            comparison_stats.append({
                "Phase": phase_name,
                "AUTO samples": auto_count,
                "MANUAL samples": manual_count,
                "Matching samples": matching_count
            })

        comparison_df = pd.DataFrame(comparison_stats)

        # Overall metric
        st.metric("Overall Agreement", f"{accuracy:.2f}%")

        # Table
        st.dataframe(comparison_df, use_container_width=True)

        # Mismatch visualization
        mismatch = (df["auto_phase"] != df["phase"]).astype(int)

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            x=df["timestamp"],
            y=mismatch,
            mode="lines",
            name="Mismatch"
        ))

        fig_compare.update_layout(
            title="Auto vs Manual Differences Over Time",
            yaxis=dict(
                tickvals=[0, 1],
                 ticktext=["Match", "Mismatch"]
            )
        )

        st.plotly_chart(fig_compare, use_container_width=True)

    # EXPORT

    st.subheader("Export")

    st.download_button("Download AUTO CSV",
                       df.assign(phase=df["auto_phase"]).to_csv(index=False),
                       "auto.csv")

    st.download_button("Download MANUAL CSV",
                       df.to_csv(index=False),
                       "manual.csv")
