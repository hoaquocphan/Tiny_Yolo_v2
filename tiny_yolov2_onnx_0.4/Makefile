WORK=/home/kurata/Projects/VLP64_101_G2E_V2M/work/
SDK_ONNX=/opt/poky/2.4.3/sysroots/aarch64-poky-linux/usr/

onnxruntime: tiny_yolo_inference_sample.cpp
	${CXX} -std=c++14 tiny_yolo_inference_sample.cpp box.cpp image.cpp \
	-DONNX_ML \
	-I ${SDK_ONNX}src/debug/onnxruntime/0.4.0-r0/git/include/onnxruntime/core/session/ \
	-I ${SDK_ONNX}src/debug/onnxruntime/0.4.0-r0/git/cmake/external/onnx/ \
	-L ${SDK_ONNX}lib64/onnx \
	-L ${SDK_ONNX}lib64/external/protobuf/cmake \
	-L ${SDK_ONNX}lib64/external/re2 \
	-I ${WORK}build/tmp/sysroots-components/aarch64/stb/usr/include/ \
	-lonnxruntime_session \
	-lonnxruntime_providers \
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
	-ljpeg \
	-lpthread -O2 -fopenmp -ldl ${LDFLAGS} -o tiny_yolo_inference_sample

clean:
	rm -rf *.o onnxruntime_inference_example
