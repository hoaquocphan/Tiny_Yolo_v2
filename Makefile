WORK=/data2/hoaphan/onnxruntime/
SDK_ONNX=/data2/hoaphan/onnxruntime/sdk/sysroots/

onnxruntime: tiny_yolo_v2.cpp
	${CXX} -std=c++14 tiny_yolo_v2.cpp box.cpp image.cpp \
	-DONNX_ML \
	-I /data/jpeglib/ \
	-I /media/sf_Ubuntu/Tiny_Yolo_v2/ \
	-L /data/onnxruntime/build/Linux/RelWithDebInfo/ \
	-L /data/onnxruntime/build/Linux/RelWithDebInfo/onnx/ \
	-L /data/onnxruntime/build/Linux/RelWithDebInfo/external/protobuf/cmake/ \
	-L /data/onnxruntime/build/Linux/RelWithDebInfo/external/re2/ \
	-L /data/onnxruntime/cmake/ \
	-L /usr/lib/x86_64-linux-gnu/ \
	-lonnxruntime_session \
	-lonnxruntime_providers \
	-lautoml_featurizers \
	-lonnxruntime_framework \
	-lonnxruntime_optimizer \
	-lonnxruntime_graph \
	-lonnxruntime_common \
	-lonnx_proto \
	-lprotobuf \
	-lre2 \
	-lonnxruntime_util \
	-lonnxruntime_mlas \
	-lonnx \
	-ljpeg -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_photo -lopencv_imgcodecs \
	-lpthread -O2 -fopenmp -ldl ${LDFLAGS} -o tiny_yolo_v2

clean:
	rm -rf *.o tiny_yolo_v2
