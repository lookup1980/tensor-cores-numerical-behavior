NVCC = nvcc -g -G 

all: test-V100 test-T4 test-A100 test-H100

test-V100: tc_test_numerics-V100.cu
	$(NVCC) -o $@ -arch=sm_70 -std=c++11 $<

test-T4: tc_test_numerics-T4-A100-binary16.cu
	$(NVCC) -o $@ -arch=sm_75 -std=c++11 $<

test-A100: test-A100-binary16 test-A100-bf16 test-A100-binary64 test-A100-tf32 test-A100-binary16-details

test-A100-binary16: tc_test_numerics-T4-A100-binary16.cu
	$(NVCC) -o $@ -arch=sm_80 -std=c++11 $<

test-A100-binary16-details: tc_test_numerics-T4-A100-binary16-details.cu
	$(NVCC) -o $@ -arch=sm_80 -std=c++11 $<

test-A100-%: tc_test_numerics-A100-%.cu
	$(NVCC) -o $@ -arch=sm_80 -std=c++11 $<

test-H100: test-H100-binary16 test-H100-bf16 test-H100-binary64 test-H100-tf32 test-H100-binary16-details

test-H100-binary16: tc_test_numerics-T4-A100-binary16.cu
	$(NVCC) -o $@ -arch=sm_90 -std=c++11 $<

test-H100-binary16-details: tc_test_numerics-T4-A100-binary16-details.cu
	$(NVCC) -o $@ -arch=sm_90 -std=c++11 $<

test-H100-%: tc_test_numerics-A100-%.cu
	$(NVCC) -o $@ -arch=sm_90 -std=c++11 $<

clean: clean-V100 clean-T4 clean-A100 clean-H100 clean-result

clean-V100:
	rm -f test-V100

clean-T4:
	rm -f test-T4

clean-A100:
	rm -f test-A100-*

clean-H100:
	rm -f test-H100-*

clean-result:
	rm -f result-*