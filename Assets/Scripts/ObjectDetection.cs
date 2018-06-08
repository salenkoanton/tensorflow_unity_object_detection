using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Linq;
using TensorFlow;
using System.Threading;
using System.Threading.Tasks;

public class ObjectDetection : MonoBehaviour {

    [Header("Constants")]
    private const float MIN_SCORE = .25f;
    private const int INPUT_SIZE = 224;
    private const int IMAGE_MEAN = 0;
    private const float IMAGE_STD = 255;

    [Header("Inspector Stuff")]
    public CameraImage cameraImage;
    public TextAsset labelMap;
    public TextAsset model;
    public Color objectColor;
    public Texture2D tex;

    [Header("Private member")]
    private GUIStyle style = new GUIStyle();
    private TFGraph graph;
    private TFSession session;
    private IEnumerable<CatalogItem> _catalog;
    private List<CatalogItem> items = new List<CatalogItem>();
    private List<string> markers = new List<string>() { "flag", "mini", "micky", "princess", "kylo", "yoda" };

    [Header("Thread stuff")]
    Thread _thread;
    float[] pixels;
    Color32 pixel;
    Color32[] colorPixels;
    TFTensor[] output;
    bool pixelsUpdated = false;
    bool processingImage = true;

    List<float> results = new List<float>();

	// Use this for initialization
    IEnumerator Start() {
        #if UNITY_ANDROID
        TensorFlowSharp.Android.NativeBinding.Init();
        #endif

        pixels = new float[INPUT_SIZE * INPUT_SIZE * 3];
        _catalog = CatalogUtil.ReadCatalogItems(labelMap.text);
        Debug.Log("Loading graph...");
        graph = new TFGraph();
        graph.Import(model.bytes);
        session = new TFSession(graph);
        Debug.Log("Graph Loaded!!!");
       /*foreach(var i in graph.GetEnumerator())
        {
            Debug.Log(i.Name);
        }*/
        //set style of labels and boxes
        style.fontSize = 30;
        style.contentOffset = new Vector2(0, 50);
        style.normal.textColor = objectColor;

        // Begin our heavy work on a new thread.
        _thread = new Thread(ThreadedWork);
        _thread.Start();
        //do this to avoid warnings
        processingImage = true;
        yield return new WaitForEndOfFrame();
        processingImage = false;
    }


    void ThreadedWork() {
        while (true) {
            if (pixelsUpdated) {
                Debug.Log("start");
                TFShape shape = new TFShape(1, INPUT_SIZE, INPUT_SIZE, 3);

                var tensor = TFTensor.FromBuffer(shape, pixels, 0, pixels.Length);
                var runner = session.GetRunner();

                runner.AddInput(graph["input"][0], tensor).Fetch(graph["final_result"][0]);
                Debug.Log("init");
                    output = runner.Run();

                output = runner.Run();
                Debug.Log("end");

                var result = (float[,])output[0].GetValue(false);
                
                results.Clear();
                foreach (var i in result)
                {
                    results.Add(i);
                }
                /*var boxes = (float[,,])output[0].GetValue(jagged: false);
                var scores = (float[,])output[1].GetValue(jagged: false);
                var num = (float[])output[2].GetValue(jagged: false);
                var classes = (float[,])output[3].GetValue(jagged: false);*/
                //loop through all detected objects
                /*for (int i = 0; i < result.Length; i++)
                { 
                    Debug.Log(result[i]);
                }*/
                pixelsUpdated = false;
            }
        }
    }

    IEnumerator ProcessImage(){
        colorPixels = cameraImage.ProcessImage();
        //update pixels (Cant use Color32[] on non monobehavior thread
        for (int i = 0; i < colorPixels.Length; ++i) {
            pixel = colorPixels[i];
            pixels[i * 3 + 0] = (float)((pixel.r - IMAGE_MEAN) / IMAGE_STD);
            pixels[i * 3 + 1] = (float)((pixel.g - IMAGE_MEAN) / IMAGE_STD);
            pixels[i * 3 + 2] = (float)((pixel.b - IMAGE_MEAN) / IMAGE_STD);
        }
        //flip bool so other thread will execute
        pixelsUpdated = true;
        //Resources.UnloadUnusedAssets();
        processingImage = false;
        yield return null;
    }

	private void Update() {
        if (!pixelsUpdated && !processingImage){
            processingImage = true;
            StartCoroutine(ProcessImage());
        }
        Debug.Log(pixelsUpdated + " " + processingImage);
	}

	void OnGUI() {
        try {
            int i = 0;

            foreach (float item in results) {
                Rect rect = new Rect( 10 , 10 + i++ * 50,400 , 50);

                GUI.backgroundColor = objectColor;
                //display score and label
                //GUI.Box(item.Box, item.DisplayName + '\n' + Mathf.RoundToInt(item.Score*100) + "%", style);
                //display only score
                GUI.TextField(rect, string.Format("{1} score: {0:0.0000}", item, markers[i - 1]), style);
            }
        } catch (InvalidOperationException e) {
            Debug.Log("Collection modified during Execution " + e);
        }
    }
}

