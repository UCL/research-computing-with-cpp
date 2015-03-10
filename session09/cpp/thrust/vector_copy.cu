/// "copy"

thrust::host_vector<int> H1;
thrust::host_vector<int> H2(H1);

/// "template_copy"

thrust::host_vector<float> H(5);
thrust::host_vector<double> D(H);

/// "transfer_copy"

thrust::host_vector<int> H;
thrust::device_vector<int> D(H);

/// "stl_copy"

std::vector<int> stl_vector;
thrust::device_vector<int> D(stl_vector);
