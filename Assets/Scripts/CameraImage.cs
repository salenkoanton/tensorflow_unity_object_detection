using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class CameraImage : MonoBehaviour {

    WebCamTexture webcamTexture;
    [SerializeField]
    RawImage image;
    [SerializeField]
    AspectRatioFitter aspectRatio;

    void Start() {
        //delay initialize camera
        webcamTexture = new WebCamTexture();
        image.texture = webcamTexture;
        webcamTexture.Play();
    }

    private void Update()
    {
        aspectRatio.aspectRatio = (float)webcamTexture.height / (float)webcamTexture.width;
    }

    public Color32[] ProcessImage(){
        //crop
        var cropped = TextureTools.CropTexture(webcamTexture);

        //scale
        var scaled = TextureTools.scaled(cropped, 224, 224, FilterMode.Bilinear);
        //run detection
        return scaled.GetPixels32();
    }
}
