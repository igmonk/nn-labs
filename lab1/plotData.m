function plotData(X, y)

  % Create New Figure
  figure;
  hold on;

  % 'k+' for the positive examples.
  % 'ko' for the negative examples.

  % Find indices of positive examples
  pos = find(y == 1);

  % Find indeces of negative examples
  neg = find(y == 0);

  % Plot examples
  plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
  plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)

  hold off;

end
