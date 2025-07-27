import numpy as np

try:
    import my_project
    print("Successfully imported my_project")
    
    # Check if medium_ad exists
    print(f"medium_ad exists: {'medium_ad' in dir(my_project)}")
    print(f"medium_ad_batch exists: {'medium_ad_batch' in dir(my_project)}")
    print(f"MediumAdStream exists: {'MediumAdStream' in dir(my_project)}")
    
    # Test basic functionality
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = my_project.medium_ad(data, 5)
    print(f"\nTest data: {data}")
    print(f"Result: {result}")
    print(f"Result length: {len(result)}")
    
    # Test batch
    batch_result = my_project.medium_ad_batch(data, (3, 5, 1))
    print(f"\nBatch result shape: {batch_result['values'].shape}")
    print(f"Batch periods: {batch_result['periods']}")
    
    # Test streaming
    stream = my_project.MediumAdStream(5)
    print("\nStreaming test:")
    for val in data[:7]:
        result = stream.update(val)
        print(f"  Update({val}) -> {result}")
    
    print("\nAll tests passed!")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()