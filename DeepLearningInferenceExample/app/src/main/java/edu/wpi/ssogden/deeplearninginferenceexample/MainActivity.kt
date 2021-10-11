package edu.wpi.ssogden.deeplearninginferenceexample

import android.content.Context
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import edu.wpi.ssogden.deeplearninginferenceexample.ml.LiteModelImagenetMobilenetV3Small075224Classification5Metadata1
import org.tensorflow.lite.support.image.TensorImage
import android.content.res.AssetManager
import android.widget.Button
import android.widget.ImageView
import android.widget.Switch
import android.widget.Toast
import org.checkerframework.checker.nullness.qual.NonNull
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.system.measureNanoTime
import kotlin.system.measureTimeMillis

const val USE_GPU : Boolean = false
const val NUM_SAMPLES : Int = 1


class MainActivity : AppCompatActivity() {
    private val TAG: String? = "MainActivity"
    lateinit var gpu_interpreter:Interpreter
    var gpu_interpreter_initilized = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        var bitmap = getBitmapFromAsset(this, "test_image.jpg")
        this.findViewById<ImageView>(R.id.imageInferenceDisplay).setImageBitmap(bitmap)

        val run_button = findViewById(R.id.buttonRunInference) as Button
        run_button.setOnClickListener{
            val bitmap_resized = Bitmap.createScaledBitmap(bitmap!!, 224, 224, false)
            var run_type_bool = findViewById<Switch>(R.id.switchInferenceTechnique).isChecked
            var response = ""
            var timeInMillis = measureTimeMillis {
                for (i in 0 until NUM_SAMPLES) {
                    if (run_type_bool) {
                        response = runInference_automatic(this, bitmap_resized!!)
                    } else {
                        response = runInference_manual(this, bitmap_resized!!)
                    }
                }
            }
            Log.d(TAG, "average inference executed in ${timeInMillis/1000.0}ms")
            Toast.makeText(this@MainActivity, "\"${response}\" in ${timeInMillis/1000.0}ms", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onPostResume() {
        Log.i(TAG, "onPostResume called")
        super.onPostResume()
        var bitmap = getBitmapFromAsset(this, "test_image.jpg")

        var response : String = "unknown"
        if (bitmap != null) {
            var timeInMillis = measureTimeMillis {
                response = runInference_automatic(this, Bitmap.createScaledBitmap(bitmap!!, 224, 224, false))
            }
            Log.d(TAG,"Inference (auto) took: ${timeInMillis}ms (${response})")
        }

        if (bitmap != null) {
            var timeInMillis = measureTimeMillis {
                response = runInference_manual(this, Bitmap.createScaledBitmap(bitmap!!, 224, 224, false))
            }
            Log.d(TAG,"Inference (manual) took: ${timeInMillis}ms (${response})")
        }

    }

    fun runInference_automatic(context: Context, bitmap: Bitmap): String {
        Log.i(TAG, "runInference_automatic called")
        val model = LiteModelImagenetMobilenetV3Small075224Classification5Metadata1.newInstance(context)

        // Creates inputs for reference.
        val image = TensorImage.fromBitmap(bitmap)

        // Runs model inference and gets result.
        val outputs = model.process(image)
        val logit = outputs.logitAsCategoryList

        model.close()

        val idx_of_best = logit.indices.maxByOrNull { logit[it].score }
        val response = logit[idx_of_best!!].label

        // Releases model resources if no longer used.
        return response
    }

    fun runInference_manual(context: Context, bitmap: Bitmap): String {
        Log.i(TAG, "runInference_manual called")
        var model_path = "mobilenet.tflite"
        var num_threads = 4

        val dic = loadDictionary(context.assets, "labels.txt")

        var interpreter = getInterpreter(context, model_path, num_threads)
        interpreter.allocateTensors()

        var input_buffer = convertBitmapToByteBuffer(bitmap)

        var output_buffer_size = 1001 // num_categories
        //var output_buffer_size = interpreter.getOutputTensor(0).numBytes()
        var output_buffer = FloatBuffer.allocate(output_buffer_size)


        val inputs = arrayOf(input_buffer)
        val outputs = mapOf(0 to output_buffer)
        interpreter.runForMultipleInputsOutputs(inputs, outputs)

        var output_arr = output_buffer.array()

        if (! gpu_interpreter_initilized) {
            interpreter.close()
        }
        return dic[output_arr.indices.maxByOrNull { output_buffer[it] }].toString();

    }


    private fun getInterpreter(context: Context, model_path: String, num_threads: Int): Interpreter {

        var model = loadModelFile(context.assets, model_path)

        val options = Interpreter.Options()
        val compatList = CompatibilityList()

        if (USE_GPU && compatList.isDelegateSupportedOnThisDevice) {
            if ( ! gpu_interpreter_initilized) {
                // if the device has a supported GPU, add the GPU delegate
                val delegateOptions = compatList.bestOptionsForThisDevice
                val gpuDelegate = GpuDelegate(delegateOptions)
                options.addDelegate(gpuDelegate)
                gpu_interpreter = Interpreter( model!!, options)
                //gpu_interpreter_initilized = true
            }
            return gpu_interpreter
        } else {
            // if the GPU is not supported, run on given number of threads
            options.setNumThreads(num_threads)
        }
        return Interpreter(model!!, options)
    }


    fun getBitmapFromAsset(context: Context, filePath: String?): Bitmap? {
        val assetManager: AssetManager = context.getAssets()
        val istr: InputStream
        var bitmap: Bitmap? = null
        try {
            istr = assetManager.open(filePath!!)
            bitmap = BitmapFactory.decodeStream(istr)
        } catch (e: IOException) {
            // handle exception
            Log.e(TAG, e.toString())
        }
        return bitmap
    }


    @Throws(IOException::class)
    fun loadModelFile(assetManager: AssetManager, model_path: String?): MappedByteBuffer? {
        assetManager.openFd(model_path!!).use { fileDescriptor ->
            FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
                val fileChannel = inputStream.channel
                val startOffset = fileDescriptor.startOffset
                val declaredLength = fileDescriptor.declaredLength
                return fileChannel.map(
                    FileChannel.MapMode.READ_ONLY,
                    startOffset,
                    declaredLength
                )
            }
        }
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val width = 224 // Determined by model
        val height = 224 // Determined by model
        val num_channels = 3 // Determined by bitmap
        val byteBuffer = ByteBuffer.allocateDirect(width * height * num_channels * 4) // WIDTH * HEIGHT * PIXEL_SIZE
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(width * height)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until width) {
            for (j in 0 until height) {
                val `val` = intValues[pixel++]
                byteBuffer.putFloat( ((`val` shr 16 and 0xFF) / 255.0f) ) // Channel 1
                byteBuffer.putFloat( ((`val` shr 8 and 0xFF) / 255.0f) ) // Channel 2
                byteBuffer.putFloat( ((`val` and 0xFF) / 255.0f) ) // Channel 3
            }
        }
        return byteBuffer
    }

    fun loadDictionary(
        assetManager: AssetManager,
        labels_path: String
    ): MutableMap<Int, String> {
        var dic: MutableMap<Int, String> = HashMap()

        var reader = BufferedReader(
            InputStreamReader(getAssets().open(labels_path),
                "UTF-8"))

        var curr_idx = 0
        for (line in reader.readLines()) {
            dic.put(curr_idx++, line)
        }
        return dic
    }
}