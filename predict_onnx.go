package main

import (
	"log"
	//"image"

	"github.com/ivansuteja96/go-onnxruntime"
	"github.com/disintegration/imaging"
)

const (
	//test_image_path = "data/a3.png"
	test_image_path = "data/aimg.jpg"

	fas2Model1_path = "convert2onnx/outputs/2.7_80x80_MiniFASNetV2.onnx"
	fas2Model2_path = "convert2onnx/outputs/4_0_0_80x80_MiniFASNetV1SE.onnx"

	fas_model_input_size = 80
)

var (
	fas2Model1 *onnxruntime.ORTSession
	fas2Model2 *onnxruntime.ORTSession
)

func main() {
	if err := loadModelWeight(); err!=nil {
		log.Fatal("Load model fail: ", err.Error())
	}

	// load image
	srcImage, err := imaging.Open(test_image_path)
	if err != nil {
		log.Fatal("Error: %s\n", err.Error())
	}

	shape1 := []int64{1, 3, fas_model_input_size, fas_model_input_size}
	input1 := preprocessImage(srcImage, fas_model_input_size)

	//log.Println(input1[:100])

	// 模型1 检测
	res, err := fas2Model1.Predict([]onnxruntime.TensorValue{
		{
			Value: input1,
			Shape: shape1,
		},
	})
	if err != nil {
		log.Println(err)
		return
	}

	if len(res) == 0 {
		log.Println("Failed get result")
		return
	}

	predictionA1 := res[0].Value.([]float32)
	predictionB1 := softmax(predictionA1)

	log.Println("predictionA1:", predictionA1)
	log.Println("predictionB1:", predictionB1)



	// 模型2 检测
	res2, err := fas2Model2.Predict([]onnxruntime.TensorValue{
		{
			Value: input1,
			Shape: shape1,
		},
	})
	if err != nil {
		log.Println(err)
		return
	}

	if len(res2) == 0 {
		log.Println("Failed get result")
		return
	}

	predictionA2 := res2[0].Value.([]float32)
	predictionB2 := softmax(predictionA2)

	log.Println("predictionA2:", predictionA2)
	log.Println("predictionB2:", predictionB2)

	log.Println("Real Score: ", (predictionB1[1]+predictionB2[1])/2 )

}


func loadModelWeight() (err error) {
	ortEnvDet := onnxruntime.NewORTEnv(onnxruntime.ORT_LOGGING_LEVEL_ERROR, "development")
	ortDetSO := onnxruntime.NewORTSessionOptions()

	fas2Model1, err = onnxruntime.NewORTSession(ortEnvDet, fas2Model1_path, ortDetSO)
	if err != nil {
		return err
	}

	fas2Model2, err = onnxruntime.NewORTSession(ortEnvDet, fas2Model2_path, ortDetSO)
	if err != nil {
		return err
	}

	return nil
}
