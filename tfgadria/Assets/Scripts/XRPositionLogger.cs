using System;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.InputSystem.XR;

public class XRPositionLogger : MonoBehaviour
{
    [Header("Tracked Transforms (auto-detected if left empty)")]
    public Transform headset;
    public Transform leftController;
    public Transform rightController;

    [Header("Logging Settings")]
    [Tooltip("How often to record a sample (in seconds). 0 = every frame.")]
    public float sampleInterval = 0.01f;

    [Tooltip("Print to Unity console every N samples (0 = disabled).")]
    public int consolePrintEveryN = 100;

    [Tooltip("Enable CSV file logging.")]
    public bool enableFileLog = true;

    private StreamWriter _writer;
    private string _filePath;
    private float _lastSampleTime;
    private int _sampleCount;
    private StringBuilder _sb = new StringBuilder(512);
    private bool _controllersFound;

    // Static tracking state — other scripts (e.g. UDPMarkerReceiver) check this
    // to know when to start logging OptiTrack data
    private static bool _isTrackingActive;
    private static float _trackingStartTime;
    public static bool IsTrackingActive => _isTrackingActive;
    public static float TrackingStartTime => _trackingStartTime;

    void Start()
    {
        _isTrackingActive = false;
        _trackingStartTime = -1f;
        FindTransforms();
        SetupCSV();
        _lastSampleTime = Time.time;
        _sampleCount = 0;
    }

    void Update()
    {
        if (!_controllersFound)
            FindControllers();

        if (sampleInterval > 0 && Time.time - _lastSampleTime < sampleInterval)
            return;

        _lastSampleTime = Time.time;

        float t = Time.time;

        Vector3 hPos = headset != null ? headset.position : Vector3.zero;
        Vector3 hRot = headset != null ? headset.eulerAngles : Vector3.zero;
        Vector3 lcPos = leftController != null ? leftController.position : Vector3.zero;
        Vector3 lcRot = leftController != null ? leftController.eulerAngles : Vector3.zero;
        Vector3 rcPos = rightController != null ? rightController.position : Vector3.zero;
        Vector3 rcRot = rightController != null ? rightController.eulerAngles : Vector3.zero;

        // Detect real tracking: headset X or Z moves from 0, or rotation changes
        if (!_isTrackingActive)
        {
            bool hasRealData = Mathf.Abs(hPos.x) > 0.001f
                            || Mathf.Abs(hPos.z) > 0.001f
                            || Mathf.Abs(hRot.x) > 0.1f
                            || Mathf.Abs(hRot.y) > 0.1f
                            || Mathf.Abs(hRot.z) > 0.1f;
            if (!hasRealData) return;

            _isTrackingActive = true;
            _trackingStartTime = t;
            Debug.Log($"[XRPositionLogger] Tracking active at t={t:F4}s. Logging started.");
        }

        _sampleCount++;

        RecordCSVRow(t, hPos, hRot, lcPos, lcRot, rcPos, rcRot);

        if (consolePrintEveryN > 0 && _sampleCount % consolePrintEveryN == 0)
        {
            _sb.Clear();
            _sb.Append("[XRPositionLogger] t=").Append(t.ToString("F2")).Append("s");
            _sb.Append(" | Head: ").Append(Vec3Str(hPos));
            _sb.Append(" | L-Ctrl: ").Append(Vec3Str(lcPos));
            _sb.Append(" | R-Ctrl: ").Append(Vec3Str(rcPos));
            Debug.Log(_sb.ToString());
        }
    }

    void OnDestroy()
    {
        ShutdownWriter();
        Debug.Log("[XRPositionLogger] Stopped. " + _sampleCount + " samples recorded.");
    }

    void OnApplicationQuit()
    {
        ShutdownWriter();
    }

    private void FindTransforms()
    {
        if (headset == null)
        {
            Camera cam = Camera.main;
            if (cam != null) headset = cam.transform;
        }

        FindControllers();

        Debug.Log("[XRPositionLogger] Headset: " + (headset != null ? headset.name : "NOT FOUND")
            + " | L-Ctrl: " + (leftController != null ? leftController.name : "NOT FOUND")
            + " | R-Ctrl: " + (rightController != null ? rightController.name : "NOT FOUND"));
    }

    private void FindControllers()
    {
        if (leftController == null)
        {
            var drivers = FindObjectsByType<TrackedPoseDriver>(FindObjectsSortMode.None);
            foreach (var driver in drivers)
            {
                string name = driver.gameObject.name;
                if (name == "Left Controller")
                    leftController = driver.transform;
                else if (name == "Right Controller")
                    rightController = driver.transform;
            }
        }

        _controllersFound = leftController != null && rightController != null;

        if (_controllersFound && _sampleCount == 0)
        {
            Debug.Log("[XRPositionLogger] Controllers found: L=" + leftController.name
                + " R=" + rightController.name);
        }
    }

    private void SetupCSV()
    {
        if (!enableFileLog) return;

        string folder = Path.Combine(Application.dataPath, "..", "Positions", "VR");

        if (!Directory.Exists(folder))
            Directory.CreateDirectory(folder);

        string ts = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        _filePath = Path.Combine(folder, "xr_positions_" + ts + ".csv");

        _writer = new StreamWriter(_filePath, false, Encoding.UTF8);
        _writer.AutoFlush = true;
        _writer.WriteLine(
            "time,"
            + "head_pos_x,head_pos_y,head_pos_z,"
            + "head_rot_x,head_rot_y,head_rot_z,"
            + "lctrl_pos_x,lctrl_pos_y,lctrl_pos_z,"
            + "lctrl_rot_x,lctrl_rot_y,lctrl_rot_z,"
            + "rctrl_pos_x,rctrl_pos_y,rctrl_pos_z,"
            + "rctrl_rot_x,rctrl_rot_y,rctrl_rot_z"
        );

        Debug.Log("[XRPositionLogger] CSV output: " + _filePath);
    }

    private void RecordCSVRow(float t, Vector3 hP, Vector3 hR,
                              Vector3 lcP, Vector3 lcR, Vector3 rcP, Vector3 rcR)
    {
        if (_writer == null) return;

        _sb.Clear();
        _sb.Append(t.ToString("F4"));
        AppendVec(_sb, hP); AppendVec(_sb, hR);
        AppendVec(_sb, lcP); AppendVec(_sb, lcR);
        AppendVec(_sb, rcP); AppendVec(_sb, rcR);
        _writer.WriteLine(_sb.ToString());
    }

    private static void AppendVec(StringBuilder sb, Vector3 v)
    {
        sb.Append(',').Append(v.x.ToString("F6"));
        sb.Append(',').Append(v.y.ToString("F6"));
        sb.Append(',').Append(v.z.ToString("F6"));
    }

    private static string Vec3Str(Vector3 v)
    {
        return "(" + v.x.ToString("F3") + ", " + v.y.ToString("F3") + ", " + v.z.ToString("F3") + ")";
    }

    private void ShutdownWriter()
    {
        if (_writer != null)
        {
            _writer.Flush();
            _writer.Close();
            _writer = null;
        }
    }
}
