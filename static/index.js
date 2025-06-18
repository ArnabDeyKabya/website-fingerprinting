function app() {
  return {
    /* This is the main app object containing all the application state and methods. */
    // The following properties are used to store the state of the application

    // results of cache latency measurements
    latencyResults: null,
    // local collection of trace data
    traceData: [],
    // Local collection of heapmap images
    heatmaps: [],
    traceStats: [],

    // Current status message
    status: "",
    // Is any worker running?
    isCollecting: false,
    // Is the status message an error?
    statusIsError: false,
    // Show trace data in the UI?
    showingTraces: false,

    // Collect latency data using warmup.js worker
    async collectLatencyData() {
      this.isCollecting = true;
      this.status = "Collecting latency data...";
      this.latencyResults = null;
      this.statusIsError = false;
      this.showingTraces = false;

      try {
        // Create a worker
        let worker = new Worker("warmup.js");

        // Start the measurement and wait for result
        const results = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });

        // Update results
        this.latencyResults = results;
        this.status = "Latency data collection complete!";

        // Terminate worker
        worker.terminate();
      } catch (error) {
        console.error("Error collecting latency data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Collect trace data using worker.js and send to backend
    async collectTraceData() {
       /* 
        * Implement this function to collect trace data.
        * 1. Create a worker to run the sweep function.
        * 2. Collect the trace data from the worker.
        * 3. Send the trace data to the backend for temporary storage and heatmap generation.
        * 4. Fetch the heatmap from the backend and add it to the local collection.
        * 5. Handle errors and update the status.
        */
         this.isCollecting = true;
      this.status = "Collecting trace data...";
      this.statusIsError = false;

      try {
        const worker = new Worker("worker.js");

        const trace = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });

        this.traceData = trace;
        this.status = "Trace collected, sending to server...";

        const res = await fetch("/collect_trace", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(trace),
        });

        const data = await res.json();
        if (res.ok) {
          this.status = "Trace uploaded and heatmap received.";
          this.heatmaps.push(data.heatmap);
          this.traceStats.push(data.stats);
          this.status += " Prediction: " + data.prediction;
        } else {
          throw new Error(data.error || "Upload failed");
        }

        worker.terminate();
      } catch (err) {
        this.status = "Error collecting trace: " + err.message;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Download the trace data as JSON (array of arrays format for ML)
    async downloadTraces() {
       /* 
        * Implement this function to download the trace data.
        * 1. Fetch the latest data from the backend API.
        * 2. Create a download file with the trace data in JSON format.
        * 3. Handle errors and update the status.
        */
        this.status = "Downloading traces...";
      try {
        const res = await fetch("/api/traces");
        const data = await res.json();
        const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = "traces.json";
        a.click();
        a.remove();

        this.status = "Download complete.";
      } catch (err) {
        this.status = "Download error: " + err.message;
        this.statusIsError = true;
      }
    },

    // Clear all results from the server
    async clearResults() {
      /* 
       * Implement this function to clear all results from the server.
       * 1. Send a request to the backend API to clear all results.
       * 2. Clear local copies of trace data and heatmaps.
       * 3. Handle errors and update the status.
       */
       this.status = "Clearing...";
      try {
        const res = await fetch("/api/clear_results", { method: "POST" });
        const data = await res.json();
        this.status = data.message;

        this.traceData = [];
        this.heatmaps = [];
        this.latencyResults = null;
      } catch (err) {
        this.status = "Error clearing: " + err.message;
        this.statusIsError = true;
      }
    },
  };
}
