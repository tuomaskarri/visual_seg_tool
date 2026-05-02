"""
ROS2 Recording + Learning-from-Demonstration Segmentation Tool

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
REQUIREMENTS

- ROS2 installed and sourced
- Python 3
- Required Python packages:
    streamlit
    pandas
    numpy
    plotly

Install missing packages with:
    pip install streamlit pandas numpy plotly

-----------------------------------------------------------
HOW TO RUN

Start the app:
    streamlit run your_script_name.py

-----------------------------------------------------------
RECORDING WORKFLOW

1. Enter a recording name
2. Click ▶️ Start recording
3. Perform your demonstration
4. Click ⏹️ Stop recording

Recorded topics:
    /hand_R/applied_forces
    /hand_R/finger_joint_angles
    /hand_R/hand_pose

-----------------------------------------------------------
CSV CONVERSION

Click:
    "Convert all rosbags to CSV"

This runs:
    convert_all_rosbags_to_csv.py

Make sure this script exists in the same directory.

-----------------------------------------------------------
SEGMENTATION WORKFLOW

1. Upload a CSV demonstration
2. Adjust parameters in sidebar:
    - Force threshold
    - Speed threshold
    - Closure threshold
3. View automatic segmentation
4. Edit segments manually if needed
5. Compare AUTO vs MANUAL

-----------------------------------------------------------
EXPORT

Download:
    - AUTO segmentation CSV
    - MANUAL segmentation CSV

-----------------------------------------------------------
NOTES

- Segmentation is based on:
    force, speed, and finger closure
- Rolling median is used for noise reduction
- Help with streamlit https://docs.streamlit.io/get-started/installation/command-line
- If your recorded demonstrations are saved in the VM’s root filesystem, they may not be accessible from Windows. To          resolve this, create a shared folder between the VM and your Windows machine, then move the demonstration files into that   shared directory. Once placed there, the files should be accessible from both environments and the workflow should          function correctly.
- A shared folder has been added that contains a couple of demonstration data files
   These recordings can be used to test the tool’s functionality (visualisation, segmentation, and comparison).

-----------------------------------------------------------
"""
