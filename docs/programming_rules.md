# Programming Rules and Conventions

## Code Organization

### 1. File Structure
- Use clear section headers with consistent formatting:
  ```python
  # =====================================================================
  # Class Name
  # =====================================================================

  # =====================================================================
  # Initialization
  # =====================================================================

  # =====================================================================
  # Private Methods
  # =====================================================================

  # =====================================================================
  # Public Methods
  # =====================================================================
  ```
- Group related functionality into separate modules
- Keep files focused and maintainable (aim for < 500 lines)
- Place utility functions in dedicated utility modules

### 2. Class Organization
Classes should be organized in the following order:
1. Class docstring with detailed description, attributes, and examples
2. Class-level constants and type definitions
3. Constructor and Initialization section
4. Private Methods section
5. Public Methods section
6. Property decorators

Example:
```python
class DataProcessor:
    """Process and analyze data sets.
    
    This class handles data loading, preprocessing, and analysis of various
    data formats. It supports multiple input types and provides methods for
    data transformation and analysis.
    
    Attributes:
        data (Optional[np.ndarray]): Current data set
        n_samples (int): Number of samples in the data set
        config (Dict[str, Any]): Processing configuration parameters
    """
    
    # Class constants
    SUPPORTED_FORMATS = ['.csv', '.json', '.parquet']
    DEFAULT_BATCH_SIZE = 1000
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the data processor with configuration."""
        self.config = config
        self.data = None
        self.n_samples = 0
        self._init_processor()
```

### 3. Method Organization
- Group related methods together with clear section comments
- Place initialization methods in the constructor section
- Place helper methods in the private methods section
- Place API methods in the public methods section
- Order methods by complexity (simple to complex)
- Keep methods focused and single-purpose (aim for < 50 lines)

## Naming Conventions

### 1. Private Methods
- All private methods must start with an underscore: `_method_name`
- Private methods should be placed in the private methods section
- Use descriptive names that indicate their purpose
- Examples:
  ```python
  def _init_processor(self) -> None:
      """Initialize processing parameters."""
      
  def _validate_input(self, file_paths: List[str]) -> None:
      """Validate input file paths and formats."""
      
  def _extract_features(self, data: np.ndarray, feature_idx: int) -> np.ndarray:
      """Extract specific features from the data set."""
  ```

### 2. Public Methods
- Public methods should not start with an underscore
- Public methods should be placed in the public methods section
- Use clear, action-oriented names
- Examples:
  ```python
  def load_dataset(self, dataset_idx: int) -> None:
      """Load a specific dataset from the collection."""
      
  def process_data(self) -> None:
      """Process all samples in the current dataset."""
      
  def extract_features(self, sample_idx: int) -> np.ndarray:
      """Extract features from a specific sample."""
  ```

### 3. Variables and Constants
- Use UPPERCASE for module-level constants
- Use snake_case for variables and functions
- Use descriptive names that indicate purpose
- Avoid single-letter names except for loop variables
- Examples:
  ```python
  MAX_ITERATIONS = 1000
  DEFAULT_THRESHOLD = 0.5
  
  def process_data(data_array: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
      for i in range(MAX_ITERATIONS):
          # Process data
  ```

## Type Hints

### 1. Method Signatures
- Use type hints for all method parameters and return values
- Use typing module for complex types
- Include type hints in docstrings for additional clarity
- Example:
  ```python
  from typing import List, Tuple, Optional, Dict, Any
  
  def process_batch(
      self,
      data_list: List[np.ndarray],
      config: Dict[str, Any]
  ) -> Tuple[List[np.ndarray], Dict[str, float]]:
      """Process a batch of data with given configuration.
      
      Args:
          data_list: List of input data arrays to process
          config: Configuration parameters for processing
          
      Returns:
          Tuple containing processed data and metrics
      """
  ```

### 2. Variable Types
- Use type hints for class attributes in docstrings
- Use Optional for nullable values
- Use Union for multiple possible types
- Example:
  ```python
  class DataProcessor:
      """Process and analyze data sets.
      
      Attributes:
          data (Optional[np.ndarray]): Current data set
          n_samples (int): Number of samples in the data set
          config (Dict[str, Any]): Processing configuration
          metrics (Dict[str, Union[float, int]]): Processing metrics
      """
  ```

## Documentation

### 1. Docstrings
- Use Google-style docstrings for all classes and methods
- Include Args, Returns, Raises sections where applicable
- Provide examples for complex methods
- Example:
  ```python
  def detect_anomalies(
      self,
      data: np.ndarray,
      min_threshold: float = 0.1
  ) -> List[Tuple[int, int, int, int]]:
      """Detect anomalies in the input data.
      
      Args:
          data: Input data as numpy array
          min_threshold: Minimum threshold for anomaly detection
          
      Returns:
          List of anomaly coordinates (x, y, width, height)
          
      Raises:
          ValueError: If data is invalid or empty
          
      Example:
          >>> processor = DataProcessor()
          >>> anomalies = processor.detect_anomalies(data, min_threshold=0.05)
      """
  ```

### 2. Comments
- Use comments to explain complex logic
- Keep comments up-to-date with code changes
- Use TODO comments for future improvements
- Example:
  ```python
  # TODO: Implement parallel processing for large datasets
  def process_large_dataset(self, data_list: List[np.ndarray]) -> None:
      # Current implementation is sequential
      # Will be optimized in future version
      for data in data_list:
          self.process_single(data)
  ```