# fas2-go 的开发原型



## 编译
```
CGO_CPPFLAGS="-I/usr/local/include/opencv4" CGO_LDFLAGS="-L/usr/local/lib -lopencv_core -lopencv_calib3d -lopencv_imgproc" go build -o data/
```



## 运行
```
LD_LIBRARY_PATH=/usr/local/lib data/onnx_test
```
