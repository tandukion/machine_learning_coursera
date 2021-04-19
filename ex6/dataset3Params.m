function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Create lists of C and sigma to test
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';

% Matrix for error tests
Error = zeros(size(C_vec,1),size(sigma_vec,1));

% Calculate error for all C and sigma
for i=1:size(C_vec,1)
    for j=1:size(sigma_vec,1)
        model = svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j)));
        predictions = svmPredict(model, Xval);

        % Get the error for current C and sigma
        Error(i,j) = mean(double(predictions ~= yval));
    endfor
endfor

% Find min error
% Normal [a b] = min(A) return [colMin rowIndex]
[sigma_min_vec C_min_ix_vec] = min(Error);

% Get the index for sigma from colMin
[_ sigma_min_ix] = min(sigma_min_vec);

% Get the index for C from rowIndex for colMin
C_min_ix = C_min_ix_vec(sigma_min_ix);

% Return the value for C and sigma from the minimum index
C = C_vec(C_min_ix);
sigma = sigma_vec(sigma_min_ix);

% =========================================================================

end
