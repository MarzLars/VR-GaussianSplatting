using UnityEngine;
using System.Collections;
using System;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine.Profiling;


public class InputForce : MonoBehaviour
{
    
    public InputDevice _rightController;
    public InputDevice _leftController;
    public InputDevice _HMD;

    public GameObject _head;
    public GameObject _handL;
    public GameObject _handR;
    public GameObject _rayL;
    public GameObject _rayR;
    public GameObject _gs;
    public float controllerRadius = 0.4f;
    public List<InputDevice> inputDevices;

    // Start is called before the first frame update
    void Start()
    {
        //inputDevices = new List<UnityEngine.XR.InputDevice>();
        //_head = GameObject.Find("Main Camera");
        //_handL = GameObject.Find("Left Controller");
        //_handR = GameObject.Find("Right Controller");
    }

    // Update is called once per frame
    void Update()
    {
        inputDevices = new List<UnityEngine.XR.InputDevice>();
        UnityEngine.XR.InputDevices.GetDevices(inputDevices);

        if (!_rightController.isValid || !_leftController.isValid || !_HMD.isValid)
        {
            InitializeInputDevices();
        }

        _rayL.GetComponent<XRInteractorLineVisual>().lineLength = controllerRadius;
        _rayR.GetComponent<XRInteractorLineVisual>().lineLength = controllerRadius;

        var headpos = _head.transform.position;     //worldposition 
        var handLpos = _handL.transform.position;
        var handLrot = _handL.transform.rotation;
        var handRpos = _handR.transform.position;
        var handRrot = _handR.transform.rotation;
        
        if (_leftController.TryGetFeatureValue(CommonUsages.primaryButton, out bool LaButtonPressed) && LaButtonPressed)
        {
            _gs.GetComponent<GaussianSplatting>().SavePly();
        }
        if (_rightController.TryGetFeatureValue(CommonUsages.primaryButton, out bool aButtonPressed) && aButtonPressed)
        {
            Debug.Log("A");
            controllerRadius = Mathf.Max(0.05f, controllerRadius - 0.05f);
        }
        if (_rightController.TryGetFeatureValue(CommonUsages.secondaryButton, out bool bButtonPressed) && bButtonPressed)
        {
            Debug.Log("B");
            controllerRadius = Mathf.Min(2.0f, controllerRadius + 0.05f);
        }

        bool Ltriggered = _leftController.TryGetFeatureValue(CommonUsages.triggerButton, out bool _Ltriggered) && _Ltriggered;
        bool Rtriggered = _rightController.TryGetFeatureValue(CommonUsages.triggerButton, out bool _Rtriggered) && _Rtriggered;
        _gs.GetComponent<GaussianSplatting>().SetController(
            controllerRadius,
            handLpos, handLrot, Ltriggered ? 1 : 0,
            handRpos, handRrot, Rtriggered ? 1 : 0);

        UnityEngine.XR.InputDevices.GetDevices(inputDevices);
    }

    //input stroke smoothing and resampling
    private void InitializeInputDevices()
    {

        if (!_rightController.isValid)
            InitializeInputDevice(InputDeviceCharacteristics.Controller | InputDeviceCharacteristics.Right, ref _rightController);
        if (!_leftController.isValid)
            InitializeInputDevice(InputDeviceCharacteristics.Controller | InputDeviceCharacteristics.Left, ref _leftController);
        if (!_HMD.isValid)
            InitializeInputDevice(InputDeviceCharacteristics.HeadMounted, ref _HMD);

    }

        private void InitializeInputDevice(InputDeviceCharacteristics inputCharacteristics, ref InputDevice inputDevice)
    {
        List<InputDevice> devices = new List<InputDevice>();
        //Call InputDevices to see if it can find any devices with the characteristics we're looking for
        InputDevices.GetDevicesWithCharacteristics(inputCharacteristics, devices);

        //Our hands might not be active and so they will not be generated from the search.
        //We check if any devices are found here to avoid errors.
        if (devices.Count > 0)
        {
            inputDevice = devices[0];
        }
    }


        public Vector3[] Kalman(Vector3[] _positions, float R, float Q)
    {
        float[] x = new float[_positions.Length];
        float[] y = new float[_positions.Length];
        float[] z = new float[_positions.Length];
        for (int i = 0; i < _positions.Length; i++)
        {
            x[i] = _positions[i].x;
            y[i] = _positions[i].y;
            z[i] = _positions[i].z;
        }
        x = Filter(x, R, Q);
        y = Filter(y, R, Q);
        z = Filter(z, R, Q);
        for (int i = 0; i < _positions.Length; i++)
        {
            _positions[i].x = x[i];
            _positions[i].y = y[i];
            _positions[i].z = z[i];
        }
        return _positions;
    }
    
    public float[] Filter(float[] z, float R, float Q)
    {
        float[] xhat = new float[z.Length];
        xhat[0] = z[0];
        float P = 1;

        for (int k = 1; k < xhat.Length; k++)
        {
            float xhatminus = xhat[k - 1];
            float Pminus = P + Q;
            float K = Pminus / (Pminus + R);
            xhat[k] = xhatminus + K * (z[k] - xhatminus);
            P = (1 - K) * Pminus;
        }
        return xhat;
    }


    //impltmented resampling from:  A novel framework for making dominant point detection methods non-parametric.
    public void _Resample(Vector3[] positions, int start, int end, float threshold)
    {
        int mid = (start + end) / 2;
        Vector3 line1 = positions[mid] - positions[start];
        Vector3 line2 = positions[end] - positions[mid];
        for (int i = start + 1; i < mid; i++)
        {
            var dist = Vector3.Cross(line1, positions[i] - positions[start]).magnitude;
            if (dist > threshold)
            {
                _Resample(positions, start, mid, threshold);
                break;
            }
            else if (i == mid - 1)
            {
                return;
            }
        }
        for (int i = mid + 1; i < end; i++)
        {
            var dist = Vector3.Cross(line2, positions[i] - positions[mid]).magnitude;
            if (dist > threshold)
            {
                _Resample(positions, mid, end, threshold);
                break;
            }
            else if (i == end - 1)
            {
                return;
            }
        }
        return;
    }
}
