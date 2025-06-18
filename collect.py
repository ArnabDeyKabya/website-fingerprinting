import time
import json
import os
import signal
import sys
import random
import traceback
import socket
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import database
from database import Database

WEBSITES = [
    # websites of your choice
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com",
    "https://prothomalo.com",
]

TRACES_PER_SITE = 1000
FINGERPRINTING_URL = "http://localhost:5000" 
OUTPUT_PATH = "dataset.json"

# Initialize the database to save trace data reliably
database.db = Database(WEBSITES)

""" Signal handler to ensure data is saved before quitting. """
def signal_handler(sig, frame):
    print("\nReceived termination signal. Exiting gracefully...")
    try:
        database.db.export_to_json(OUTPUT_PATH)
    except:
        pass
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


"""
Some helper functions to make your life easier.
"""

def is_server_running(host='127.0.0.1', port=5000):
    """Check if the Flask server is running."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def setup_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    # If you want headless:
    # chrome_options.add_argument("--headless")

    # === HARD-CODE or use an environment variable for the path to your manually downloaded chromedriver.exe ===
    chromedriver_path = r"E:\Academic\Level 4-1\CSE 406 - Security Sessional\Offline 2\2005112\driver\chromedriver-win64\chromedriver.exe"
    if not os.path.isfile(chromedriver_path):
        raise FileNotFoundError(f"ChromeDriver not found at {chromedriver_path}\n"
                                "Please download the correct Windows 64-bit driver and update this path.")

    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver



def retrieve_traces_from_backend(driver):
    """Retrieve traces from the backend API."""
    traces = driver.execute_script("""
        return fetch('/api/get_results')
            .then(response => response.ok ? response.json() : {traces: []})
            .then(data => data.traces || [])
            .catch(() => []);
    """)
    
    count = len(traces) if traces else 0
    print(f"  - Retrieved {count} traces from backend API" if count else "  - No traces found in backend storage")
    return traces or []

def clear_trace_results(driver, wait):
    """Clear all results from the backend by pressing the button."""
    clear_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Clear all results')]")
    clear_button.click()

    wait.until(EC.text_to_be_present_in_element(
        (By.XPATH, "//div[@role='alert']"), "Cleared"))
    
def is_collection_complete():
    """Check if target number of traces have been collected."""
    current_counts = database.db.get_traces_collected()
    remaining_counts = {website: max(0, TRACES_PER_SITE - count) 
                      for website, count in current_counts.items()}
    return sum(remaining_counts.values()) == 0

"""
Your implementation starts here.
"""

def collect_single_trace(driver, wait, website_url):
    """ Implement the trace collection logic here. 
    1. Open the fingerprinting website
    2. Click the button to collect trace
    3. Open the target website in a new tab
    4. Interact with the target website (scroll, click, etc.)
    5. Return to the fingerprinting tab and close the target website tab
    6. Wait for the trace to be collected
    7. Return success or failure status
    """
    print(f"Collecting trace for: {website_url}")

    try:
        # Open the tracing frontend
        driver.get(FINGERPRINTING_URL)
        time.sleep(1)

        # Start trace collection
        collect_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Collect Trace')]")
        collect_btn.click()
        time.sleep(0.5)

        # Open the target site in a new tab
        driver.execute_script(f"window.open('{website_url}', '_blank');")
        driver.switch_to.window(driver.window_handles[1])
        try:
            wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
        except:
            print(f"[!] Timeout waiting for {website_url} to fully load. Continuing anyway...")

        time.sleep(8)  # allow page load

        # Simulate scroll
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # Close tab, return to trace site
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        time.sleep(3)  # allow trace data to be captured

        # Get the latest trace from backend
        traces = retrieve_traces_from_backend(driver)
        if traces:
            trace = traces[-1]  # Use the most recent trace
            website_index = WEBSITES.index(website_url)
            saved = database.db.save_trace(website_url, website_index, trace)
            return saved
        else:
            print("[!] No trace received.")
            return False
    except Exception as e:
        print(f"[!] Error collecting trace: {e}")
        print(traceback.format_exc())  # Show full details of error
        return False

def collect_fingerprints(driver, target_counts=None):
    """ Implement the main logic to collect fingerprints.
    1. Calculate the number of traces remaining for each website
    2. Open the fingerprinting website
    3. Collect traces for each website until the target number is reached
    4. Save the traces to the database
    5. Return the total number of new traces collected
    """
    wait = WebDriverWait(driver, 10)
    if target_counts is None:
        target_counts = {site: TRACES_PER_SITE for site in WEBSITES}

    while not is_collection_complete():
        for website in WEBSITES:
            current_count = database.db.get_traces_collected().get(website, 0)
            if current_count >= target_counts[website]:
                continue

            success = collect_single_trace(driver, wait, website)
            if not success:
                print(f"[-] Failed to collect trace for {website}")
            else:
                print(f"[+] Trace collected for {website} ({current_count + 1}/{target_counts[website]})")

    print("[✓] Collection complete.")

def main():
    """ Implement the main function to start the collection process.
    1. Check if the Flask server is running
    2. Initialize the database
    3. Set up the WebDriver
    4. Start the collection process, continuing until the target number of traces is reached
    5. Handle any exceptions and ensure the WebDriver is closed at the end
    6. Export the collected data to a JSON file
    7. Retry if the collection is not complete
    """
    if not is_server_running():
        print("❌ Flask server is not running at localhost:5000. Please start it first.")
        return

    print("✅ Flask server is up. Starting collection...")

    database.db.init_database()

    driver = setup_webdriver()
    try:
        collect_fingerprints(driver)
        database.db.export_to_json(OUTPUT_PATH)
    except Exception as e:
        print(f"[!] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        driver.quit()
        print("[✓] WebDriver closed.")


if __name__ == "__main__":
    main()
