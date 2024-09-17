// Original maps
const maps = [
    [[1, 2], [3, 4]],
    [[2, 3], [4, 5]],
    [[3, 4], [5, 6]]
  ];
  
  // Helper function to calculate mean
  const mean = arr => arr.reduce((a, b) => a + b) / arr.length;
  
  // Helper function to calculate standard deviation
  const std = arr => {
    const m = mean(arr);
    return Math.sqrt(arr.reduce((a, b) => a + Math.pow(b - m, 2), 0) / arr.length);
  };
  
  // Step 1: Normalize maps
  const normalizedMaps = maps.map(m => {
    const flat = m.flat();
    const m_mean = mean(flat);
    const m_std = std(flat);
    return m.map(row => row.map(cell => (cell - m_mean) / (m_std + 1e-8)));
  });
  
  console.log("Normalized Maps:", normalizedMaps);
  
  // Step 2: Calculate similarity matrix
  const flatMaps = normalizedMaps.map(m => m.flat());
  const cosineSimilarity = (a, b) => {
    const dotProduct = a.reduce((sum, _, i) => sum + a[i] * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, x) => sum + x * x, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, x) => sum + x * x, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  };
  const simMatrix = flatMaps.map(m1 => flatMaps.map(m2 => cosineSimilarity(m1, m2) + 1));
  
  console.log("Similarity Matrix:", simMatrix);
  
  // Step 3: Calculate weights
  const weights = simMatrix.map(row => mean(row));
  
  console.log("Weights:", weights);
  
  // Step 4: Weighted aggregation
  const weightedMaps = normalizedMaps.map((m, i) => m.map(row => row.map(cell => cell * weights[i])));
  const aggregatedMap = weightedMaps[0].map((row, i) => 
    row.map((_, j) => weightedMaps.reduce((sum, m) => sum + m[i][j], 0) / sum(weights))
  );
  
  console.log("Aggregated Map:", aggregatedMap);