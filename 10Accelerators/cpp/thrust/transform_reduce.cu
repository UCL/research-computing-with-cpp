thrust::device_vector<int> d(5)
thrust::negate<float> unary_op; 
thrust::plus<float> binary_op; 
float init = 0;

thrust::transform_reduce(d.begin(), d.end(), unary_op, init, binary_op)