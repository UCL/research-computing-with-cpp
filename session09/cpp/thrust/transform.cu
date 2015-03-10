thrust::device_vector<int> X(10); 
thrust::device_vector<int> Y(10); 

// initialize X to 0,1,2,3, .... 
thrust::sequence(X.begin(), X.end()); 

//compute Y = -X 
// thrust::transform(first, last, result, unary_op);
thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<int>());