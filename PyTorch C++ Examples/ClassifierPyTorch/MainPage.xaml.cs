using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;

using Windows.AI.MachineLearning;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.Storage.Streams;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Media.Imaging;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace ClassifierPyTorch
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private ImageClassifierModel modelGen;
        private ImageClassifierInput image = new ImageClassifierInput();
        private ImageClassifierOutput results;
        private StorageFile selectedStorageFile;
        private string label = "";
        private float probability = 0;
        private Helper helper = new Helper();

        public enum Labels
        {
            plane,
            car,
            bird,
            cat,
            deer,
            dog,
            frog,
            horse,
            ship,
            truck
        }

        // A method to evaluate the model.
        private async Task evaluate()
        {
            results = await modelGen.EvaluateAsync(image);
        }

        // A method to extract output from the model.
        private void extractResults()
        {
            // Retrieve the results of evaluation.
            var mResult = results.modelOutput as TensorFloat;

            // Convert the result to vector format.
            var resultVector = mResult.GetAsVectorView();

            float probability = float.MinValue;
            int index = 0;

            // Find the maximum probability.
            for (int i = 0; i < resultVector.Count; i++)
            {
                float elementProbability = (float)resultVector[i];

                if (elementProbability > probability)
                {
                    index = i;
                    probability = elementProbability;
                }

                System.Diagnostics.Debug.WriteLine(i + " " + elementProbability);
            }

            label = ((Labels)index).ToString();
        }

        private async Task displayResult()
        {
            displayOutput.Text = label;
        }

        // A method to select an input image file.
        private async Task<bool> getImage()
        {
            try
            {
                // Trigger file picker to select an image file.
                FileOpenPicker fileOpenPicker = new FileOpenPicker();
                fileOpenPicker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
                fileOpenPicker.FileTypeFilter.Add(".jpg");
                fileOpenPicker.FileTypeFilter.Add(".png");
                fileOpenPicker.ViewMode = PickerViewMode.Thumbnail;
                selectedStorageFile = await fileOpenPicker.PickSingleFileAsync();

                if (selectedStorageFile == null)
                {
                    return false;
                }
            }
            catch (Exception)
            {
                return false;
            }

            return true;
        }

        // A method to convert and bide the input image.
        private async Task imageBind()
        {
            UIPreviewImage.Source = null;

            try
            {
                SoftwareBitmap softwareBitmap;
                
                using (IRandomAccessStream stream = await selectedStorageFile.OpenAsync(FileAccessMode.Read))
                {
                    // Create the decoder from the stream.
                    BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);

                    // Get the SoftwareBitmap representation of the file in BGRA8 format.
                    softwareBitmap = await decoder.GetSoftwareBitmapAsync();
                    softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
                }

                // Display the image.
                SoftwareBitmapSource imageSource = new SoftwareBitmapSource();
                await imageSource.SetBitmapAsync(softwareBitmap);
                UIPreviewImage.Source = imageSource;

                // Encapsulate the image within a VideoFrame to be bound and evaluated.
                VideoFrame inputImage = VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);

                // Resize the image size to 32x32.
                inputImage = await helper.CropAndDisplayInputImageAsync(inputImage);

                // Bind the model input with image.
                ImageFeatureValue imageTensor = ImageFeatureValue.CreateFromVideoFrame(inputImage);
                image.modelInput = imageTensor;
            }
            catch (Exception e)
            {
            }
        }

        // Waiting for a click event to select a file.
        private async void OpenFileButton_Click(object sender, RoutedEventArgs e)
        {
            if (!await getImage())
            {
                return;
            }

            // After the click event happened and an input selected, begin the model execution.
            // Bind the model input.
            await imageBind();

            // Model evaluation.
            await evaluate();

            // Extract the results.
            extractResults();

            // Display the results.
            await displayResult();
        }

        private async Task loadModel()
        {
            // Get an access to the ONNX model and save it in memory.
            StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/ImageClassifier.onnx"));

            // Instantiate the model.
            modelGen = await ImageClassifierModel.CreateFromStreamAsync(modelFile);
        }

        public MainPage()
        {
            this.InitializeComponent();

            loadModel();
        }
    }
}
