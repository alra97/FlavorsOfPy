import streamlit as st
import time
import multiprocessing
import concurrent.futures
import numpy as np
import cProfile

# Fibonacci recursive function (for testing)
def fibonacci_recursive(n):
    if n <= 1:
        return n
    else:
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# Optimized Fibonacci using dynamic programming
def fibonacci_dynamic(n):
    fib = [0, 1]
    for i in range(2, n+1):
        fib.append(fib[i-1] + fib[i-2])
    return fib[n]

# Function to measure execution time and profile
def profile_function(func, *args):
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args)
    pr.disable()
    pr.print_stats()
    return result

# Performance Comparison Section
def performance_comparison():
    st.title("Performance Comparison of Python Implementations")
    st.markdown("""
    ### Comparing the Performance of Different Python Implementations
    We'll compare CPython, PyPy, and Jython using the Fibonacci sequence.
    - **CPython**: The default implementation.
    - **PyPy**: Python with JIT compilation.
    - **Jython**: Python for Java Virtual Machine.
    
    You can run these tests on your system with the respective Python flavors.
    """)
    
    n = st.number_input("Enter the Fibonacci number for computation:", min_value=20, max_value=40, value=30)
    
    st.write("Running test with CPython (or your default Python implementation)...")
    start = time.time()
    fibonacci_recursive(n)
    end = time.time()
    st.write(f"Execution Time with CPython: {end - start:.4f} seconds")

    st.markdown("Run the same test with PyPy or Jython for comparison.")

# Function to calculate Fibonacci numbers for a given range
def fib_worker(start, end):
    result = []
    for i in range(start, end):
        result.append(fibonacci_recursive(i))
    return result

# Parallelization of Fibonacci Algorithm
def parallel_fibonacci(n, workers=2):
    # Split the range of Fibonacci numbers into workers
    chunk_size = n // workers
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(workers)]
    
    # Make sure the last worker gets the remaining tasks
    if ranges[-1][1] < n:
        ranges[-1] = (ranges[-1][0], n)
    
    # Parallelize the Fibonacci computation
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(fib_worker, [r[0] for r in ranges], [r[1] for r in ranges]))
    
    # Flatten the list of results
    return [item for sublist in results for item in sublist]

def parallelization_section():
    st.header("Parallelization of Fibonacci Algorithm")
    st.markdown("""
    In this part, we'll parallelize the Fibonacci sequence calculation and measure performance improvement.
    """)
    
    n = st.number_input("Enter the Fibonacci number for computation (parallelized):", min_value=20, max_value=40, value=30)
    workers = st.slider("Select the number of parallel workers:", 1, multiprocessing.cpu_count(), 2)

    if st.button("Run Parallel Test"):
        st.write("Running parallel Fibonacci computation...")
        start = time.time()
        parallel_fibonacci(n, workers)
        end = time.time()
        st.write(f"Execution Time with {workers} workers: {end - start:.4f} seconds")

# Big O Analysis of Fibonacci
def big_o_analysis():
    st.header("Big O Analysis of Fibonacci Algorithm")
    st.markdown("""
    The Fibonacci algorithm has a time complexity of **O(2^n)** for the recursive version.
    In the parallelized version, we aim to reduce the execution time, but the Big O notation remains **O(2^n)**.
    
    Let's compare the execution times of both recursive and parallel Fibonacci algorithms.
    """)
    
    n = st.number_input("Enter the Fibonacci number for Big O analysis:", min_value=20, max_value=150, value=30)
    
    if st.button("Run Recursive Fibonacci"):
        st.write("Running recursive Fibonacci...")
        start = time.time()
        fibonacci_recursive(n)
        end = time.time()
        st.write(f"Execution Time (Recursive, O(2^n)): {end - start:.4f} seconds")
    
    if st.button("Run Dynamic Programming Fibonacci"):
        st.write("Running dynamic programming Fibonacci...")
        start = time.time()
        fibonacci_dynamic(n)
        end = time.time()
        st.write(f"Execution Time (Dynamic Programming, O(n)): {end - start:.4f} seconds")

# Streamlit App Layout
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Performance Comparison", "Parallelization", "Big O Analysis"])

if page == "Introduction":
    st.title("Fibonacci Algorithm Performance Evaluation")
    st.markdown("""
    This application compares the performance of different Python implementations (CPython, PyPy, Jython)
    using the Fibonacci sequence, as well as evaluates the impact of parallelization on execution speed.
    """)
elif page == "Performance Comparison":
    performance_comparison()
elif page == "Parallelization":
    parallelization_section()
elif page == "Big O Analysis":
    big_o_analysis()
