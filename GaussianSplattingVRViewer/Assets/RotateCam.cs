using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RotateCam : MonoBehaviour
{
    // Start is called before the first frame update
    private float radius;
    private float initialAngle;
    private Vector3 initialPosition;
    private int frm = 0;

    void Start()
    {
        initialPosition = new Vector3(0, 0.3f, -1.8f);
        radius = Mathf.Sqrt(initialPosition.x * initialPosition.x + initialPosition.z * initialPosition.z);

        initialAngle = Mathf.Atan2(initialPosition.z, initialPosition.x);
    }

    // Update is called once per frame
    void Update()
    {
        frm += 1;
        float cycleLength = 10.0f;
        float cyclePosition = ((frm / 37.0f) % cycleLength) * 1.0f / cycleLength;

        float angleOffset = Mathf.Sin(cyclePosition * 2 * Mathf.PI) * Mathf.PI / 4;

        float currentAngle = initialAngle + angleOffset;
        float x = radius * Mathf.Cos(currentAngle);
        float z = radius * Mathf.Sin(currentAngle);

        transform.position = new Vector3(x, initialPosition.y, z);

        transform.LookAt(new Vector3(0, initialPosition.y, 0));
    }
}
