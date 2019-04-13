function [J, grad] = costFunction(theta, X, y)

  m = length(y); % number of training examples
  X = [ones(1, m); X];
  h = theta * X; % [1 m]
  diff = h - y;
  square = (diff .^ 2);
  J = (0.5 / m) * sum(sum(square));

  grad = (1 / m) * (X * diff');

end;
