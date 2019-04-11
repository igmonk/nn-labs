function [J, grad] = costFunction(theta, X, y)

  % Initialization
  m = length(y); % number of training examples
  J = 0;
  grad = zeros(size(theta));

  h = sigmoid(X * theta); % size(X) = [100 3]; size(theta) = [3 1];
  term1 = (-y) .* log(h);
  term2 = (1 - y) .* log(1 - h);
  J = sum(term1 - term2) / m;

  grad = (1 / m) * (X' * (h - y));

end
