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


C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

predict_error = 1000; % why not predict_error = 0 ???

for i = 1: size(C_values,2)
    
    C_i = C_values(i);
    
    for j = 1: size(sigma_values,2)
        
        sigma_j = sigma_values(j);
        
        model = svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_j));
        
        predictions = svmPredict(model, Xval);
        
        p_error = mean(double(predictions ~= yval));
        
        if p_error < predict_error % why not p_error > predict_error ???
            
            fprintf('\nPrediction error is %f with C %f and sigma %f', p_error, C_i, sigma_j);
            
            predict_error = p_error;
            
            C = C_i;
            
            sigma = sigma_j;

        end
    end
end




% =========================================================================

end
