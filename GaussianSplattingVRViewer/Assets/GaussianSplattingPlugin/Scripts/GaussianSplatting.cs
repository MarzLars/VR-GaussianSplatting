using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.Assertions;

public class GaussianSplatting : MonoBehaviour
{

    private class GaussianSplattingNI
    {
        public const int INIT_EVENT = 0x0001;
        public const int DRAW_EVENT = 0x0002;

        [DllImport("gaussiansplatting", EntryPoint = "GetRenderEventFunc")] public static extern System.IntPtr GetRenderEventFunc();
        [DllImport("gaussiansplatting", EntryPoint = "IsAPIReady")] public static extern bool IsAPIReady();
        [DllImport("gaussiansplatting", EntryPoint = "GetLastMessage")] private static extern System.IntPtr _GetLastMessage();
        static public string GetLastMessage() { return Marshal.PtrToStringAnsi(_GetLastMessage()); }
        [DllImport("gaussiansplatting", EntryPoint = "LoadModel")] public static extern bool LoadModel(string file);
        [DllImport("gaussiansplatting", EntryPoint = "SetNbPov")] public static extern void SetNbPov(int nb_pov);
        [DllImport("gaussiansplatting", EntryPoint = "SetPovParameters")] public static extern void SetPovParameters(int pov, int width, int height);
        [DllImport("gaussiansplatting", EntryPoint = "IsInitialized")] public static extern bool IsInitialized();
        [DllImport("gaussiansplatting", EntryPoint = "GetTextureNativePointer")] public static extern System.IntPtr GetTextureNativePointer(int pov);
        [DllImport("gaussiansplatting", EntryPoint = "SetDrawParameters")] public static extern void SetDrawParameters(int pov, float[] position, float[] rotation, float[] proj, float fovy, float[] frustums);
        [DllImport("gaussiansplatting", EntryPoint = "SetLightingParameters")] public static extern void SetLightingParameters(int H, int W, float[] position, float[] rotation, float[] proj, float fovy, float[] frustums);
        [DllImport("gaussiansplatting", EntryPoint = "SetSimParams")] public static extern void SetSimParams(
            float frame_dt, 
            float dt, 
            float gravity,
            float damping_coeffient, 
            int rest_iter, 
            float collision_stiffness, 
            float minimal_dist, 
            float collision_detection_iter_interval,
            float shadow_eps,
            float shadow_factor,
            int zup, 
            float ground_height, 
            float global_scale, 
            float[] global_offset, 
            float[] global_rotation, 
            int num_objects, 
            float[] object_offsets, 
            float[] object_materials,
            int disable_LG_interp,
            int quasi_static,
            int max_quasi_sequence,
            int enable_export_frames,
            int max_export_sequence,
            int boundary);
        [DllImport("gaussiansplatting", EntryPoint = "SetController")] public static extern void SetController(
            float controller_radius,
            float[] _lpos, float[] _lrot, int _ltriggered,
            float[] _rpos, float[] _rrot, int _rtriggered);
        [DllImport("gaussiansplatting", EntryPoint = "GetSceneSize")] public static extern void GetSceneSize(float[] scene_min, float[] scene_max);
        [DllImport("gaussiansplatting", EntryPoint = "IsDrawn")] public static extern bool IsDrawn();
        [DllImport("gaussiansplatting", EntryPoint = "GetNbSplat")] public static extern int GetNbSplat();
        [DllImport("gaussiansplatting", EntryPoint = "Save")] public static extern void Save();
    }

    [Header("Init Parameters")]
    public string ModelPath = "C:/Users/g1n0st/OneDrive/Desktop/GaussianSplattingVRViewerUnity/Data/garden/";
    public Material mat;
    public Camera cam;
    public bool isXr;

    [Header("Dynamic Parameters")]
    public bool loadModelEvent = false;
    public bool sendInitEvent = false;
    public bool sendDrawEvent = false;
    [Range(0.1f, 1f)]
    public float texFactor = 0.5f;

    [Header("Informations")]
    public bool loaded = false;
    public bool initialized = false;
    public int nb_splats = 0;
    public string lastMessage = "";
    public Vector2Int internalTexSize;
    public Texture2D[] tex;
    public bool isInError = false;
    public Vector3 sceneMin = Vector3.zero;
    public Vector3 sceneMax = Vector3.one;

    private float lastTexFactor = 0.5f;
    private GameObject real_leye, real_reye;
    private System.IntPtr renderEventFunc = System.IntPtr.Zero;
    private Thread thLoad = null;
    private Texture2D blackTexture = null;
    private int countDrawErrors = 0;
    private bool waitForTexture = false;

    [Header("Simulation Parameters")]
    public float frameDt = 7.5e-3f;
    public float dt = 1e-3f + 1e-5f;
    public float gravity = -4.0f;
    public float dampingCoeffient = 5.0f;
    public int XPBDRestIter = 25;
    public float collisionStiffness = 0.1f;
	public float collisionMinimalDist = 5e-3f;
    public float collisionDetectionIterInterval = 100;
    public bool isZUp = false;

    public float groundHeight = 0.0f;

    public float globalScale = 1.0f;

    public Quaternion globalRotation;

    public Vector3 globalOffset;

    public Vector3[] objectOffsets;
    public int boundary = 0;

    [System.Serializable]
    public class MaterialProperty {
        public float density;
        public float E;
        public float nu;
        public float scale = 1.0f;
        public bool isRigid = false;
        public Quaternion rot = Quaternion.identity;
    }

	public List<MaterialProperty> objectMaterials;

    [Header("Lighting Parameters")]
    public bool useShadowMap = true;
    public int shadowMapHeight = 1024;
    public int shadowMapWidth = 1024;
    public float shadowEps = 0.05f;
    public float shadowFactor = 0.5f;
    public Camera shadowMapCamera;

    [Header("Experimental Options")]
    public bool disableTwoLevelInterpolation = false;
    public bool quasiStatic = false;
    public int maxQuasiSequence = 100;
    public bool enableExportFrames = false;
    public int maxExportSequence = 100;

    bool TryGetEyesPoses(out Vector3 lpos, out Vector3 rpos, out Quaternion lrot, out Quaternion rrot)
    {
        lpos = Vector3.zero;
        rpos = Vector3.zero;
        lrot = Quaternion.identity;
        rrot = Quaternion.identity;
        int nbfound = 0;
        List<XRNodeState> states = new List<XRNodeState>();
        InputTracking.GetNodeStates(states);

        foreach (XRNodeState state in states)
        {
            if (state.tracked && state.nodeType == XRNode.LeftEye)
            {
                if (state.TryGetPosition(out Vector3 tpos)) { lpos = tpos; nbfound += 1; }
                if (state.TryGetRotation(out Quaternion trot)) { lrot = trot; nbfound += 1; }
            }
            if (state.tracked && state.nodeType == XRNode.RightEye)
            {
                if (state.TryGetPosition(out Vector3 tpos)) { rpos = tpos; nbfound += 1; }
                if (state.TryGetRotation(out Quaternion trot)) { rrot = trot; nbfound += 1; }
            }
        }
        return nbfound == 4;
    }

    private void Start()
    {
        isInError = false;
        countDrawErrors = 0;
        internalTexSize = Vector2Int.zero;
        tex = new Texture2D[isXr ? 2 : 1];
        lastTexFactor = texFactor;
        blackTexture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
        blackTexture.LoadRawTextureData(new byte[] { 0, 0, 0, 255 });
        blackTexture.Apply();
        SetBlackTexture();

        if (isZUp) {
            transform.localRotation = Quaternion.Euler(-90.0f, 0.0f, 0.0f);
        } else {
            transform.localRotation = Quaternion.Euler(0.0f, 0.0f, 0.0f);
        }
        if (!isXr) { // Debug Viewpoint
            // trackTRS.localScale = Vector3.one * 0.36f;
            // trackTRS.localPosition = new Vector3(0.0f, 0.0f, 1.5f);
        }
    }

    private void OnEnable()
    {
        Camera.onPreRender += OnPreRenderCallback;
        isInError = false;
    }

    private void OnDisable()
    {
        Camera.onPreRender -= OnPreRenderCallback;
        tex = null;
        lastMessage = "";
    }

    public void SetBlackTexture()
    {
        mat.SetTexture("_GaussianSplattingTexLeftEye", blackTexture);
        mat.SetTexture("_GaussianSplattingTexRightEye", blackTexture);
    }

    public void SetSimParams() {
        if (!loaded) {
            float[] floatObjectOffsets = new float[objectOffsets.Length * 3];
            float[] floatGlobalOffset = {globalOffset.x, globalOffset.y, globalOffset.z};
            float[] floatGlobalRotation = { globalRotation.x, globalRotation.y, globalRotation.z, globalRotation.w };
            const int propertySize = 9;
            float[] floatObjectMaterials = new float[objectMaterials.Count * propertySize];

            Assert.IsTrue(objectMaterials.Count == objectOffsets.Length);
            for (int i = 0; i < objectOffsets.Length; i++)
            {
                floatObjectOffsets[i * 3 + 0] = objectOffsets[i].x;
                floatObjectOffsets[i * 3 + 1] = objectOffsets[i].y;
                floatObjectOffsets[i * 3 + 2] = objectOffsets[i].z;

                floatObjectMaterials[i * propertySize + 0] = objectMaterials[i].density;
                floatObjectMaterials[i * propertySize + 1] = objectMaterials[i].E;
                floatObjectMaterials[i * propertySize + 2] = objectMaterials[i].nu;
                floatObjectMaterials[i * propertySize + 3] = objectMaterials[i].scale;
                floatObjectMaterials[i * propertySize + 4] = objectMaterials[i].isRigid ? 1.0f : 0.0f;

                floatObjectMaterials[i * propertySize + 5] = objectMaterials[i].rot.x;
                floatObjectMaterials[i * propertySize + 6] = objectMaterials[i].rot.y;
                floatObjectMaterials[i * propertySize + 7] = objectMaterials[i].rot.z;
                floatObjectMaterials[i * propertySize + 8] = objectMaterials[i].rot.w;
            }
            GaussianSplattingNI.SetSimParams(frameDt,
                                             dt,
                                             gravity,
                                             dampingCoeffient,
                                             XPBDRestIter,
                                             collisionStiffness,
                                             collisionMinimalDist,
                                             collisionDetectionIterInterval,
                                             shadowEps,
                                             shadowFactor,
                                             isZUp ? 1 : 0,
                                             groundHeight,
                                             globalScale,
                                             floatGlobalOffset,
                                             floatGlobalRotation,
                                             objectOffsets.Length,
                                             floatObjectOffsets,
                                             floatObjectMaterials,
                                             // Experimental Options
                                             disableTwoLevelInterpolation ? 1 : 0,
                                             quasiStatic ? 1 : 0,
                                             maxQuasiSequence,
                                             enableExportFrames ? 1 : 0,
                                             maxExportSequence,
                                             boundary
                                             );
        }
    }

    public void SetController(float controller_radius, Vector3 lpos, Quaternion lrot, int ltrigger,
                              Vector3 rpos, Quaternion rrot, int rtrigger) {
        // rot = Quaternion.identity; 
        if (loaded) {
            if (isZUp) {
                float[] _lpos = {lpos.x, lpos.z, lpos.y};
                float[] _lrot = {lrot.x, lrot.z, lrot.y, lrot.w};
                float[] _rpos = {rpos.x, rpos.z, rpos.y};
                float[] _rrot = {rrot.x, rrot.z, rrot.y, rrot.w};
                GaussianSplattingNI.SetController(
                    controller_radius,
                    _lpos, _lrot, ltrigger,
                    _rpos, _rrot, rtrigger);
            } else {
                float[] _lpos = {lpos.x, -lpos.y, lpos.z};
                float[] _lrot = {lrot.x, -lrot.y, lrot.z, lrot.w};
                float[] _rpos = {rpos.x, -rpos.y, rpos.z};
                float[] _rrot = {rrot.x, -rrot.y, rrot.z, rrot.w};
                GaussianSplattingNI.SetController(
                    controller_radius,
                    _lpos, _lrot, ltrigger,
                    _rpos, _rrot, rtrigger);
            }
        }
    }

    public void SavePly() {
        if (loaded) {
            GaussianSplattingNI.Save();
        }
    }

    public void SetLightingParameters() {
        float fovy = shadowMapCamera.fieldOfView * Mathf.PI / 180;
        Matrix4x4 proj_mat = shadowMapCamera.projectionMatrix;
        Vector3 pos = shadowMapCamera.transform.position;
        Quaternion rot = shadowMapCamera.transform.rotation;
        pos = transform.InverseTransformPoint(pos);
        rot = Quaternion.Inverse(transform.rotation) * rot;

        rot = Quaternion.Euler(0, 0, 180) * Quaternion.Euler(rot.eulerAngles.x, -rot.eulerAngles.y, -rot.eulerAngles.z);
        pos.y = -pos.y;

        FrustumPlanes decomp = proj_mat.decomposeProjection;
        float[] position = { pos.x, pos.y, pos.z };
        float[] rotation = { rot.x, rot.y, rot.z, rot.w };
        float[] proj = matToFloat(proj_mat);
        float[] planes = { decomp.left, decomp.right, decomp.bottom, decomp.top, decomp.zNear, decomp.zFar };

        GaussianSplattingNI.SetLightingParameters(shadowMapHeight, shadowMapWidth, position, rotation, proj, fovy, planes);
    }

    private void Update()
    {
        //If thread is finished set it to null
        if (thLoad != null && thLoad.Join(0))
        {
            thLoad = null;
        }

        //Wait for xr ready
        if (isXr && !TryGetEyesPoses(out Vector3 _lpos, out Vector3 _rpos, out Quaternion _lrot, out Quaternion _rrot))
        {
            return;
        }

        if (lastTexFactor != texFactor)
        {
            sendInitEvent = true;
        }

        if (!Directory.Exists(ModelPath))
        {
            lastMessage = "Directory '" + ModelPath + "' does not exists.";
            isInError = true;
            return;
        }

        if (GaussianSplattingNI.IsAPIReady())
        {
            initialized = GaussianSplattingNI.IsInitialized();
            lastMessage = GaussianSplattingNI.GetLastMessage();

            if (loadModelEvent)
            {
                loadModelEvent = false;
                if (thLoad == null)
                {
                    thLoad = new Thread(() => {
                        SetSimParams();
                        loaded = GaussianSplattingNI.LoadModel(ModelPath);
                    });
                    thLoad.Start();
                }
            }

            if (renderEventFunc == System.IntPtr.Zero)
            {
                renderEventFunc = GaussianSplattingNI.GetRenderEventFunc();
            }

            if (loaded && renderEventFunc != System.IntPtr.Zero)
            {
                if (sendInitEvent)
                {
                    sendInitEvent = false;
                    isInError = false;
                    countDrawErrors = 0;

                    GaussianSplattingNI.SetNbPov(isXr ? 2 : 1);
                    internalTexSize = new Vector2Int((int)((float)cam.pixelWidth * texFactor), (int)((float)cam.pixelHeight * texFactor));
                    
                    for (int i = 0; i < (isXr ? 2 : 1); ++i)
                    {
                        //Set plugins parameters for pov
                        GaussianSplattingNI.SetPovParameters(i, internalTexSize.x, internalTexSize.y);
                    }
                    lastTexFactor = texFactor;

                    GL.IssuePluginEvent(renderEventFunc, GaussianSplattingNI.INIT_EVENT);

                    //Now loading is separated from init so we can wait end of initialization.
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    while (!GaussianSplattingNI.IsInitialized() && sw.ElapsedMilliseconds < 100000)
                    {
                        Thread.Sleep(0);
                    }
                    
                    initialized = GaussianSplattingNI.IsInitialized();

                    if (sw.ElapsedMilliseconds >= 100000)
                    {
                        lastMessage = GaussianSplattingNI.GetLastMessage();
                        Debug.Log("Stop Waiting for init end: " + lastMessage);
                        isInError = true;
                    }
                    else
                    {
                        //Init done get external texture
                        for (int i = 0; i < (isXr ? 2 : 1); ++i)
                        {
                            IntPtr texPtr = GaussianSplattingNI.GetTextureNativePointer(i);
                            tex[i] = Texture2D.CreateExternalTexture(internalTexSize.x, internalTexSize.y, TextureFormat.RGBAFloat, false, true, texPtr);

                            mat.SetTexture(i == 0 ? "_GaussianSplattingTexLeftEye" : "_GaussianSplattingTexRightEye", tex[i]);
                        }
                    }
                }

                if (sendDrawEvent)
                {
                    waitForTexture = false;
                    bool doit = true;
                    if (isXr)
                    {
                        if (TryGetEyesPoses(out Vector3 lpos, out Vector3 rpos, out Quaternion lrot, out Quaternion rrot))
                        {
                            if (real_leye == null) { real_leye = new GameObject("real leye"); real_leye.transform.parent = cam.transform.parent; }
                            real_leye.transform.localPosition = lpos;
                            real_leye.transform.localRotation = lrot;

                            if (real_reye == null) { real_reye = new GameObject("real reye"); real_reye.transform.parent = cam.transform.parent; }
                            real_reye.transform.localPosition = rpos;
                            real_reye.transform.localRotation = rrot;
                        }
                        else
                        {
                            doit = false;
                        }
                    }

                    if (doit)
                    {
                        // Shadow map light source setting
                        if (useShadowMap) {
                            SetLightingParameters();
                        }

                        for (int i = 0; i < (isXr ? 2 : 1); ++i)
                        {
                            if (tex[i] != null)
                            {
                                float fovy = cam.fieldOfView * Mathf.PI / 180;
                                Matrix4x4 proj_mat = cam.projectionMatrix;
                                Vector3 pos = cam.transform.position;
                                Quaternion rot = cam.transform.rotation;

                                if (isXr)
                                {
                                    if (i == 0)
                                    {
                                        proj_mat = cam.GetStereoProjectionMatrix(Camera.StereoscopicEye.Left);
                                        pos = real_leye.transform.position;
                                        rot = real_leye.transform.rotation;
                                    }
                                    else
                                    {
                                        proj_mat = cam.GetStereoProjectionMatrix(Camera.StereoscopicEye.Right);
                                        pos = real_reye.transform.position;
                                        rot = real_reye.transform.rotation;
                                    }

                                }

                                pos = transform.InverseTransformPoint(pos);
                                rot = Quaternion.Inverse(transform.rotation) * rot;

                                //TODO: Move that in dll
                                rot = Quaternion.Euler(0, 0, 180) * Quaternion.Euler(rot.eulerAngles.x, -rot.eulerAngles.y, -rot.eulerAngles.z);
                                pos.y = -pos.y;

                                FrustumPlanes decomp = proj_mat.decomposeProjection;
                                float[] position = { pos.x, pos.y, pos.z };
                                float[] rotation = { rot.x, rot.y, rot.z, rot.w };
                                float[] proj = matToFloat(proj_mat);
                                float[] planes = { decomp.left, decomp.right, decomp.bottom, decomp.top, decomp.zNear, decomp.zFar };

                                GaussianSplattingNI.SetDrawParameters(i, position, rotation, proj, fovy, planes);
                            }
                            else
                            {
                                doit = false;
                            }
                        }
                    }

                    if (doit)
                    {
                        GL.IssuePluginEvent(renderEventFunc, GaussianSplattingNI.DRAW_EVENT);
                        GL.InvalidateState();
                        waitForTexture = true;
                    }
                }
            }

            if (loaded)
            {
                nb_splats = GaussianSplattingNI.GetNbSplat();
                float[] scene_min = { 0.0f, 0.0f, 0.0f };
                float[] scene_max = { 0.0f, 0.0f, 0.0f };
                GaussianSplattingNI.GetSceneSize(scene_min, scene_max);
                sceneMin.x = scene_min[0];
                sceneMin.y = scene_min[1];
                sceneMin.z = scene_min[2];
                sceneMax.x = scene_max[0];
                sceneMax.y = scene_max[1];
                sceneMax.z = scene_max[2];
            }
        }
    }

    void OnPreRenderCallback(Camera camera) {
        if (loaded && renderEventFunc != System.IntPtr.Zero && sendDrawEvent && waitForTexture)
        {
            waitForTexture = false;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            while (!GaussianSplattingNI.IsDrawn() && sw.ElapsedMilliseconds < 100000)
            {
                Thread.Sleep(0);
            }

            if (sw.ElapsedMilliseconds >= 100000)
            {
                countDrawErrors += 1;
                //if 5 consecutive try in error stop !!!
                if (countDrawErrors >= 5)
                {
                    lastMessage = GaussianSplattingNI.GetLastMessage();
                    Debug.Log("Stop draw error: " + lastMessage);
                    isInError = true;
                    //Stop trying...
                    sendDrawEvent = false;
                }
            } else {
                countDrawErrors = 0;
            }
        }
    }

    float[] matToFloat(Matrix4x4 mat)
    {
        return new float[16]
        {
            mat.m00, mat.m10, mat.m20, mat.m30,
            mat.m01, mat.m11, mat.m21, mat.m31,
            mat.m02, -mat.m12, mat.m22, mat.m32,
            mat.m03, mat.m13, mat.m23, mat.m33,
        };
    }
}
