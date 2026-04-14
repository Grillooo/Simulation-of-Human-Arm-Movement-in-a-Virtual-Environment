using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// First-person virtual arm driven by 3 OptiTrack markers (shoulder, elbow, hand).
///
/// Press O to activate after calibration.
///
/// If armModelPrefab is assigned (drag Y Bot FBX), a realistic skinned arm is shown.
/// Otherwise falls back to capsule/sphere primitives.
///
/// Marker identification: shoulder = closest to headset, hand = farthest from shoulder.
/// Marker loss: joints hold last position; arm hides after markerLossTimeout with 0 markers.
/// </summary>
public class ArmModelController : MonoBehaviour
{
    [Header("Activation")]
    public Key activateKey = Key.O;

    [Header("Arm Model")]
    [Tooltip("Drag the Y Bot FBX (or any Mixamo humanoid) here. Leave empty for primitive fallback.")]
    public GameObject armModelPrefab;

    [Header("Skin")]
    public Color skinColor = new Color(0.87f, 0.75f, 0.65f);

    [Header("Fallback Primitives")]
    public float armRadius = 0.04f;
    public float handRadius = 0.05f;
    public Color upperArmColor = new Color(0.85f, 0.72f, 0.6f);
    public Color forearmColor  = new Color(0.82f, 0.69f, 0.57f);
    public Color handColor     = new Color(0.88f, 0.75f, 0.63f);

    [Header("Smoothing")]
    public float smoothSpeed = 20f;

    [Header("Marker Loss")]
    public float markerLossTimeout = 2.0f;

    [Header("Debug")]
    public bool showJointSpheres = true;

    [Header("Status (read-only)")]
    [SerializeField] private bool armModeActive = false;
    [SerializeField] private int visibleMarkers = 0;

    public bool IsArmModeActive => armModeActive;

    private const int SHOULDER = 0, ELBOW = 1, HAND = 2;

    private UDPMarkerReceiver _receiver;
    private CoordinateSynchronizer _synchronizer;
    private Transform _headset;

    private Vector3[] _jointTargets  = new Vector3[3];
    private Vector3[] _jointSmoothed = new Vector3[3];
    private float[]   _jointLastSeen = new float[3];
    private bool  _initialized;
    private float _allLostTimer;
    private List<Vector3> _markerPositions = new List<Vector3>();

    // ─── Rigged model state ─────────────────────────────────────────
    private bool _useModel;
    private GameObject _modelInstance;
    private Transform _upperArmBone;   // mixamorig:RightArm
    private Transform _forearmBone;    // mixamorig:RightForeArm
    private Transform _handBone;       // mixamorig:RightHand
    private Vector3    _shoulderBindOffset;
    private Quaternion _upperArmBindLocalRot;
    private Quaternion _forearmBindLocalRot;

    // ─── Fallback primitives ────────────────────────────────────────
    private GameObject _primUpperArm, _primForearm, _primHand;

    // ─── Joint debug spheres ────────────────────────────────────────
    private GameObject _shoulderSphere, _elbowSphere, _handSphere;

    // ================================================================
    //  Lifecycle
    // ================================================================

    void Start()
    {
        _receiver     = FindFirstObjectByType<UDPMarkerReceiver>();
        _synchronizer = FindFirstObjectByType<CoordinateSynchronizer>();

        Camera cam = Camera.main;
        if (cam != null) _headset = cam.transform;

        if (_receiver == null) Debug.LogError("[Arm] UDPMarkerReceiver not found!");
        if (_headset  == null) Debug.LogError("[Arm] Main Camera not found!");

        // Try rigged model first, fall back to primitives
        if (armModelPrefab != null)
        {
            _useModel = SetupRiggedModel();
            if (!_useModel)
                Debug.LogWarning("[Arm] Bone search failed — using primitive shapes.");
        }
        if (!_useModel)
            CreatePrimitives();

        CreateJointSpheres();
        SetVisualsActive(false);
    }

    void Update()
    {
        if (Keyboard.current != null && Keyboard.current[activateKey].wasPressedThisFrame)
            ToggleArmMode();

        if (!armModeActive || _receiver == null || _headset == null) return;

        // ── Gather markers, transform to Unity space ────────────────
        Dictionary<int, Vector3> allRaw = _receiver.GetAllRawPositions();
        _markerPositions.Clear();
        foreach (var kvp in allRaw)
            _markerPositions.Add(TransformPoint(kvp.Value));

        // Keep only 3 closest to headset
        if (_markerPositions.Count > 3)
        {
            _markerPositions.Sort((a, b) =>
                Vector3.SqrMagnitude(a - _headset.position)
                    .CompareTo(Vector3.SqrMagnitude(b - _headset.position)));
            _markerPositions.RemoveRange(3, _markerPositions.Count - 3);
        }

        visibleMarkers = _markerPositions.Count;
        float now = Time.time;

        // ── Marker identification / tracking ────────────────────────
        if (_markerPositions.Count >= 3)
        {
            _allLostTimer = 0f;
            if (!_initialized)
            {
                IdentifyByHeadset(_markerPositions);
                for (int j = 0; j < 3; j++)
                {
                    _jointSmoothed[j] = _jointTargets[j];
                    _jointLastSeen[j] = now;
                }
                _initialized = true;
            }
            else
            {
                MatchMarkersToJoints(_markerPositions, now);
                for (int j = 0; j < 3; j++)
                    _jointLastSeen[j] = now;
            }
        }
        else if (_markerPositions.Count > 0 && _initialized)
        {
            _allLostTimer = 0f;
            MatchMarkersToJoints(_markerPositions, now);
        }
        else if (_initialized)
        {
            _allLostTimer += Time.deltaTime;
            if (_allLostTimer > markerLossTimeout)
            {
                SetVisualsActive(false);
                return;
            }
        }
        else
        {
            SetVisualsActive(false);
            return;
        }

        // ── Smooth ──────────────────────────────────────────────────
        float t = smoothSpeed > 0 ? Time.deltaTime * smoothSpeed : 1f;
        for (int j = 0; j < 3; j++)
            _jointSmoothed[j] = Vector3.Lerp(_jointSmoothed[j], _jointTargets[j], t);

        // ── Render ──────────────────────────────────────────────────
        SetVisualsActive(true);

        if (_useModel)
            UpdateRiggedModel();
        else
            UpdatePrimitives();

        UpdateJointSpheres();
    }

    // ================================================================
    //  Marker identification (unchanged)
    // ================================================================

    private void IdentifyByHeadset(List<Vector3> markers)
    {
        Vector3 headPos = _headset.position;

        int sIdx = 0;
        float minD = float.MaxValue;
        for (int i = 0; i < 3; i++)
        {
            float d = Vector3.SqrMagnitude(markers[i] - headPos);
            if (d < minD) { minD = d; sIdx = i; }
        }

        int hIdx = -1;
        float maxD = -1f;
        for (int i = 0; i < 3; i++)
        {
            if (i == sIdx) continue;
            float d = Vector3.SqrMagnitude(markers[i] - markers[sIdx]);
            if (d > maxD) { maxD = d; hIdx = i; }
        }

        int eIdx = 3 - sIdx - hIdx;

        _jointTargets[SHOULDER] = markers[sIdx];
        _jointTargets[ELBOW]    = markers[eIdx];
        _jointTargets[HAND]     = markers[hIdx];
    }

    private void MatchMarkersToJoints(List<Vector3> markers, float now)
    {
        int mc = markers.Count;
        bool[] jointTaken = new bool[3];
        bool[] markerUsed = new bool[mc];

        for (int pass = 0; pass < Mathf.Min(mc, 3); pass++)
        {
            float bestDist = float.MaxValue;
            int bestM = -1, bestJ = -1;

            for (int m = 0; m < mc; m++)
            {
                if (markerUsed[m]) continue;
                for (int j = 0; j < 3; j++)
                {
                    if (jointTaken[j]) continue;
                    float d = Vector3.SqrMagnitude(markers[m] - _jointTargets[j]);
                    if (d < bestDist) { bestDist = d; bestM = m; bestJ = j; }
                }
            }

            if (bestM >= 0)
            {
                _jointTargets[bestJ] = markers[bestM];
                _jointLastSeen[bestJ] = now;
                jointTaken[bestJ] = true;
                markerUsed[bestM] = true;
            }
        }
    }

    private Vector3 TransformPoint(Vector3 raw)
    {
        if (_synchronizer != null && _synchronizer.IsCalibrated)
            return _synchronizer.OptiTrackToUnity(raw);
        return raw;
    }

    // ================================================================
    //  Arm mode toggle
    // ================================================================

    private void ToggleArmMode()
    {
        if (!armModeActive)
        {
            bool calibrated = _synchronizer != null && _synchronizer.IsCalibrated;
            if (!calibrated)
            {
                Debug.LogWarning("[Arm] Calibrate first (C with 4 corner markers).");
                return;
            }

            armModeActive = true;
            _initialized  = false;
            _allLostTimer = 0f;
            Debug.Log("[Arm] ON — place 3 markers on shoulder, elbow, hand.");
        }
        else
        {
            armModeActive = false;
            SetVisualsActive(false);
            Debug.Log("[Arm] OFF.");
        }
    }

    // ================================================================
    //  Rigged model setup & update
    // ================================================================

    private bool SetupRiggedModel()
    {
        _modelInstance = Instantiate(armModelPrefab, transform);
        _modelInstance.name = "ArmModel_YBot";

        // Disable Animator so we drive bones manually
        var anim = _modelInstance.GetComponent<Animator>();
        if (anim != null) anim.enabled = false;

        // Search for right arm bones by name
        foreach (var t in _modelInstance.GetComponentsInChildren<Transform>())
        {
            string n = t.name;
            if (_upperArmBone == null && n.Contains("RightArm") && !n.Contains("Fore"))
                _upperArmBone = t;
            else if (_forearmBone == null && n.Contains("RightForeArm"))
                _forearmBone = t;
            else if (_handBone == null && n.Contains("RightHand")
                     && !n.Contains("Thumb") && !n.Contains("Index")
                     && !n.Contains("Middle") && !n.Contains("Ring")
                     && !n.Contains("Pinky"))
                _handBone = t;
        }

        if (_upperArmBone == null || _forearmBone == null || _handBone == null)
        {
            Debug.LogError($"[Arm] Bone search failed — UpperArm={_upperArmBone?.name}, " +
                           $"Forearm={_forearmBone?.name}, Hand={_handBone?.name}");
            Destroy(_modelInstance);
            return false;
        }

        Debug.Log($"[Arm] Bones found: {_upperArmBone.name}, {_forearmBone.name}, {_handBone.name}");

        // Record bind-pose data
        _shoulderBindOffset    = _upperArmBone.position - _modelInstance.transform.position;
        _upperArmBindLocalRot  = _upperArmBone.localRotation;
        _forearmBindLocalRot   = _forearmBone.localRotation;

        HideNonArmParts();
        ApplySkinColor();

        return true;
    }

    private void HideNonArmParts()
    {
        // Build set of bones to KEEP (right arm chain + spine chain to root)
        HashSet<Transform> keep = new HashSet<Transform>();

        // Find RightShoulder (parent of RightArm) and keep its entire subtree
        Transform rShoulder = _upperArmBone.parent;
        if (rShoulder != null)
            AddSubtree(rShoulder, keep);

        // Keep spine chain from RightShoulder up to model root
        Transform t = rShoulder;
        while (t != null && t != _modelInstance.transform)
        {
            keep.Add(t);
            t = t.parent;
        }

        // Scale every skeleton bone NOT in the keep set to zero
        foreach (var bone in _modelInstance.GetComponentsInChildren<Transform>())
        {
            if (bone == _modelInstance.transform) continue;
            if (keep.Contains(bone)) continue;
            if (bone.GetComponent<SkinnedMeshRenderer>() != null) continue;
            if (bone.GetComponent<MeshRenderer>() != null) continue;

            // Only collapse actual skeleton bones (not root-level objects)
            if (bone.parent != _modelInstance.transform)
                bone.localScale = Vector3.zero;
        }

        // Disable shadow casting (shadow of hidden torso would look odd)
        foreach (var r in _modelInstance.GetComponentsInChildren<Renderer>())
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
    }

    private void AddSubtree(Transform root, HashSet<Transform> set)
    {
        set.Add(root);
        for (int i = 0; i < root.childCount; i++)
            AddSubtree(root.GetChild(i), set);
    }

    private void ApplySkinColor()
    {
        foreach (var r in _modelInstance.GetComponentsInChildren<Renderer>())
        {
            // Instantiate materials so we don't modify the shared asset
            Material[] mats = r.materials; // this already returns instances
            foreach (var mat in mats)
            {
                if (mat.HasProperty("_BaseColor"))
                    mat.SetColor("_BaseColor", skinColor);
                if (mat.HasProperty("_Color"))
                    mat.color = skinColor;
            }
            r.materials = mats;
        }
    }

    private void UpdateRiggedModel()
    {
        Vector3 shoulder = _jointSmoothed[SHOULDER];
        Vector3 elbow    = _jointSmoothed[ELBOW];
        Vector3 hand     = _jointSmoothed[HAND];

        // 1. Position model so upper-arm bone lands at shoulder marker
        _modelInstance.transform.position = shoulder - _shoulderBindOffset;

        // 2. Reset bones to bind pose (undo last frame's rotations)
        _upperArmBone.localRotation = _upperArmBindLocalRot;
        _forearmBone.localRotation  = _forearmBindLocalRot;

        // 3. Rotate upper arm so it points from shoulder toward elbow
        Vector3 curDir = (_forearmBone.position - _upperArmBone.position).normalized;
        Vector3 tgtDir = (elbow - shoulder).normalized;
        if (curDir.sqrMagnitude > 0.001f && tgtDir.sqrMagnitude > 0.001f)
            _upperArmBone.rotation = Quaternion.FromToRotation(curDir, tgtDir) * _upperArmBone.rotation;

        // 4. Rotate forearm so it points from elbow toward hand
        //    (forearm has moved because it's a child of upper arm)
        curDir = (_handBone.position - _forearmBone.position).normalized;
        tgtDir = (hand - elbow).normalized;
        if (curDir.sqrMagnitude > 0.001f && tgtDir.sqrMagnitude > 0.001f)
            _forearmBone.rotation = Quaternion.FromToRotation(curDir, tgtDir) * _forearmBone.rotation;
    }

    // ================================================================
    //  Fallback primitive visuals
    // ================================================================

    private void CreatePrimitives()
    {
        _primUpperArm = MakeCapsule("UpperArm", upperArmColor);
        _primForearm  = MakeCapsule("Forearm",  forearmColor);

        _primHand = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        _primHand.name = "Hand";
        _primHand.transform.SetParent(transform);
        Destroy(_primHand.GetComponent<Collider>());
        _primHand.GetComponent<Renderer>().material.color = handColor;
    }

    private GameObject MakeCapsule(string name, Color color)
    {
        GameObject g = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        g.name = name;
        g.transform.SetParent(transform);
        Destroy(g.GetComponent<Collider>());
        g.GetComponent<Renderer>().material.color = color;
        return g;
    }

    private void UpdatePrimitives()
    {
        UpdateCapsule(_primUpperArm, _jointSmoothed[SHOULDER], _jointSmoothed[ELBOW]);
        UpdateCapsule(_primForearm,  _jointSmoothed[ELBOW],    _jointSmoothed[HAND]);

        _primHand.transform.position = _jointSmoothed[HAND];
        Vector3 dir = (_jointSmoothed[HAND] - _jointSmoothed[ELBOW]).normalized;
        if (dir != Vector3.zero) _primHand.transform.forward = dir;
        float d = handRadius * 2f;
        _primHand.transform.localScale = new Vector3(d, d, d);
    }

    private void UpdateCapsule(GameObject capsule, Vector3 from, Vector3 to)
    {
        float length = Vector3.Distance(from, to);
        capsule.transform.position = (from + to) / 2f;
        capsule.transform.up = (to - from).normalized;
        float d = armRadius * 2f;
        capsule.transform.localScale = new Vector3(d, length / 2f, d);
    }

    // ================================================================
    //  Joint debug spheres
    // ================================================================

    private void CreateJointSpheres()
    {
        _shoulderSphere = MakeJointSphere("Joint_Shoulder", Color.red);
        _elbowSphere    = MakeJointSphere("Joint_Elbow",    new Color(1f, 0.7f, 0.2f));
        _handSphere     = MakeJointSphere("Joint_Hand",     Color.green);
    }

    private GameObject MakeJointSphere(string name, Color color)
    {
        GameObject g = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        g.name = name;
        g.transform.SetParent(transform);
        g.transform.localScale = Vector3.one * 0.025f;
        Destroy(g.GetComponent<Collider>());
        g.GetComponent<Renderer>().material.color = color;
        return g;
    }

    private void UpdateJointSpheres()
    {
        if (showJointSpheres)
        {
            _shoulderSphere.SetActive(true);
            _elbowSphere.SetActive(true);
            _handSphere.SetActive(true);
            _shoulderSphere.transform.position = _jointSmoothed[SHOULDER];
            _elbowSphere.transform.position    = _jointSmoothed[ELBOW];
            _handSphere.transform.position     = _jointSmoothed[HAND];
        }
    }

    // ================================================================
    //  Visibility
    // ================================================================

    private void SetVisualsActive(bool active)
    {
        if (_useModel && _modelInstance != null)
            _modelInstance.SetActive(active);

        if (!_useModel)
        {
            if (_primUpperArm != null) _primUpperArm.SetActive(active);
            if (_primForearm  != null) _primForearm.SetActive(active);
            if (_primHand     != null) _primHand.SetActive(active);
        }

        bool showJ = active && showJointSpheres;
        if (_shoulderSphere != null) _shoulderSphere.SetActive(showJ);
        if (_elbowSphere    != null) _elbowSphere.SetActive(showJ);
        if (_handSphere     != null) _handSphere.SetActive(showJ);
    }

    // ================================================================
    //  HUD
    // ================================================================

    void OnGUI()
    {
        GUIStyle style = new GUIStyle(GUI.skin.label)
        {
            fontSize  = 13,
            fontStyle = FontStyle.Bold
        };

        bool receiving  = _receiver != null && _receiver.IsReceivingData;
        bool calibrated = _synchronizer != null && _synchronizer.IsCalibrated;

        // Top-left: connection + calibration
        style.normal.textColor = receiving ? Color.green : Color.red;
        GUI.Label(new Rect(10, 10, 300, 20), receiving ? "UDP: OK" : "UDP: NO DATA", style);

        style.normal.textColor = calibrated ? Color.green : Color.yellow;
        GUI.Label(new Rect(10, 28, 300, 20), calibrated ? "CAL: OK" : "CAL: pending (C)", style);

        int count = _receiver != null ? _receiver.CurrentMarkerCount : 0;
        style.normal.textColor = Color.white;
        GUI.Label(new Rect(10, 46, 300, 20), $"Markers: {count}", style);

        string mode = _useModel ? "Model: Y Bot" : "Model: primitives";
        GUI.Label(new Rect(10, 64, 300, 20), mode, style);

        // Bottom-left: arm status
        float y = Screen.height - 30f;
        if (!armModeActive)
        {
            if (calibrated)
            {
                style.normal.textColor = Color.yellow;
                GUI.Label(new Rect(10, y, 500, 25), "Press O to activate arm mode.", style);
            }
        }
        else
        {
            string[] jn = { "Shoulder", "Elbow", "Hand" };
            float now = Time.time;

            if (visibleMarkers == 3)
            {
                style.normal.textColor = Color.green;
                GUI.Label(new Rect(10, y, 500, 25), "ARM: 3/3 tracked", style);
            }
            else if (_initialized && (visibleMarkers > 0 || _allLostTimer <= markerLossTimeout))
            {
                string s = "";
                for (int j = 0; j < 3; j++)
                {
                    float age = now - _jointLastSeen[j];
                    s += age < 0.15f ? $" {jn[j]}:OK" : $" {jn[j]}:HELD";
                }
                style.normal.textColor = Color.yellow;
                GUI.Label(new Rect(10, y, 600, 25), $"ARM: {visibleMarkers}/3 |{s}", style);
            }
            else
            {
                style.normal.textColor = Color.red;
                GUI.Label(new Rect(10, y, 400, 25), "ARM: no markers — waiting...", style);
            }
        }
    }
}
