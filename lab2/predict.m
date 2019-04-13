function p = predict(theta, X)

  m = length(X); % number of training examples
  X = [ones(1, m); X];
  p = theta * X;

end
