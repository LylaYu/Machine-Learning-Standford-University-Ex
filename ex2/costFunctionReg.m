function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
[a,n]=size(X);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%for i=1:m
%    J=J+(-y(i).*log(sigmoid(X(i,:)*theta))-(1-y(i)).*log(1-sigmoid(X(i,:)*theta)))/m;
%end
%J=J+sum(theta.^2)*lambda/2/m;
temp=(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta)))/m;
J=sum(temp)+lambda/2/m*sum(theta(2:end).^2);

for j=1:m
    grad=grad+(sigmoid(theta'*(X(j,:))')-y(j))*((X(j,:))')/m;
end
for p=2:n
    grad(p)=grad(p)+theta(p)*lambda/m;
end
% =============================================================

end
