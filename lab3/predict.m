function p = predict(Theta1, Theta2, X)

  % Useful values
  m = size(X, 1);
  num_labels = size(Theta2, 1);

  % Return the following variables correctly
  p = zeros(size(X, 1), 1);

  h0 = [ones(m, 1) X]';
  h1 = sigmoid(Theta1 * h0);
  h1 = [ones(1, m); h1];
  h2 = Theta2 * h1;
  p = h2;

end
