function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	% temporary theta variables
	theta0 = theta(1,1);
	theta1 = theta(2,1);
	
	% calculate new theta0
	A = zeros(1);
	sum0 = 0;
	for i = 1:m
		A = (theta' * X(i,:)' - y(i,:)) * X(i, 1);
		sum0 = sum0 + A(1,1);		
	end;
	sum0 = sum0 * alpha / m;
	theta0 = theta0 - sum0;

	% calculate new theta1
	B = zeros(1);
	sum1 = 0;
	for i = 1:m
		B = (theta' * X(i,:)' - y(i,:)) * X(i, 2);
		sum1 = sum1 + B(1,1);		
	end;
	sum1 = sum1 * alpha / m;
	theta1 = theta1 - sum1;
	
	% update theta0 and theta1
	theta(1,1) = theta0;
	theta(2,1) = theta1;




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);


end

end
