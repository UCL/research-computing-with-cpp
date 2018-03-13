
thrust::device_vector<int> vec(5,1);
//thrust::reduce(first, last, init, binary_op);
int sum = thrust::reduce(D.begin(), D.end(), (int) 0, thrust::plus<int>());