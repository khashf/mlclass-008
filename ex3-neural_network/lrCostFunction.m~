function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


hTheta = sigmoid(X * theta);
J = sum( (-y) .* log(hTheta) - (1.0 - y) .* log(1.0 - hTheta) );
J = J / m;
regCost = (lambda/(2.0*m)) .* sum(theta(2:end).^2.0);
J = J + regCost;

grad = (X' * (hTheta - y));
grad = grad / m;
regGrad = theta;
regGrad(1) = 0;
grad = grad + (lambda/m) .* regGrad;

