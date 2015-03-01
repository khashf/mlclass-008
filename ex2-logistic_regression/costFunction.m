function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


% Compute the cost of a particular choice of theta.
% You should set J to the cost.
% Compute the partial derivatives and set grad to the partial
% derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% compute cost
for i=1:m
	hTheta = sigmoid(theta' * X(i,:)');
	J = J + (-y(i,:) * log(hTheta) - (1 - y(i,:)) * log(1 - hTheta) );
end;
J = J / m;

% compute gradient
for i=1:size(theta,1)
	for j = 1:m
		hTheta = sigmoid(theta' * X(j,:)');
		grad(i,1) = grad(i,1) + (hTheta - y(j)) * X(j, i);
	end;
	grad(i,1) = grad(i,1) / m;
end;






% =============================================================

end
