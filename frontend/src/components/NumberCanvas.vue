<template>
  <div class="mnist-container">
    <div class="canvas-wrapper">
      <h2 class="title">Draw a Digit (0-9)</h2>
      <canvas
        ref="canvas"
        width="280"
        height="280"
        @mousedown="startPosition"
        @mouseup="endPosition"
        @mousemove="draw"
      ></canvas>
      <div class="button-group">
        <button class="clear-btn" @click="clearCanvas">Clear</button>
        <button class="predict-btn" @click="predict">Predict</button>
      </div>
    </div>
    <div class="prediction-wrapper">
      <div class="prediction-header">
        <h3>Prediction</h3>
        <Transition name="fade-out" mode="out-in">
          <div v-if="result" class="confidence-meter">
            <div class="meter-bar" :style="{ width: confidenceMeterWidth }"></div>
          </div>
        </Transition>
      </div>
      <div class="result-container">
        <Transition name="fade-out" mode="out-in">
          <div v-if="result" class="result-item">
            <span class="prediction-text">{{ result.split('(')[0] }}</span>
            <span class="confidence-text">{{ result.split('(')[1] }}</span>
          </div>
          <div v-else class="placeholder">Draw a digit and click predict...</div>
        </Transition>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      result: '',
      ctx: null,
      confidenceMeterWidth: '0%'
    }
  },
  mounted() {
    this.ctx = this.$refs.canvas.getContext('2d');
    this.ctx.fillStyle = 'white';
    this.ctx.fillRect(0, 0, this.$refs.canvas.width, this.$refs.canvas.height);
  },
  methods: {
    startPosition(e) {
      this.painting = true;
      this.draw(e);
    },
    endPosition() {
      this.painting = false;
      this.ctx.beginPath();
    },
    draw(e) {
      if (!this.painting) return;

      const rect = this.$refs.canvas.getBoundingClientRect();
      this.ctx.lineWidth = 20;
      this.ctx.lineCap = 'round';
      this.ctx.strokeStyle = 'black';
      this.ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
      this.ctx.stroke();
      this.ctx.beginPath();
      this.ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
    },
    clearCanvas() {
      this.ctx.fillStyle = 'white';
      this.ctx.fillRect(0, 0, this.$refs.canvas.width, this.$refs.canvas.height);
      this.ctx.beginPath();
      this.result = '';
      this.confidenceMeterWidth = '0%';
    },
    predict() {
      const dataURL = this.$refs.canvas.toDataURL('image/png');

      fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: dataURL })
      })
      .then(res => res.json())
      .then(data => {
        if (data.prediction !== undefined) {
          this.result = `Prediction: ${data.prediction} (Confidence: ${data.confidence})`;
          // Extract confidence percentage and set meter width
          const confidence = parseFloat(data.confidence.replace('%', ''));
          this.confidenceMeterWidth = `${confidence}%`;
        } else {
          this.result = 'Error: ' + data.error;
          this.confidenceMeterWidth = '0%';
        }
      });
    }
  }
}
</script>

<style scoped>
.mnist-container {
  display: flex;
  gap: 2rem;
  padding: 2rem;
  max-width: 1000px;
  margin: 0 auto;
  background: #f5f5f5;
  border-radius: 16px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
}

.canvas-wrapper {
  flex: 1 1 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.5rem;
}

.title {
  color: #333;
  font-size: 1.8rem;
  font-weight: 600;
  margin: 0;
  letter-spacing: 0.025em;
}

canvas {
  border: 2px solid #4CAF50;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.button-group {
  display: flex;
  gap: 1rem;
}

.clear-btn, .predict-btn {
  padding: 0.8rem 1.5rem;
  border: none;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease-out;
  min-width: 120px;
  box-shadow: 0 2px 3px rgba(0,0,0,0.1);
}

.clear-btn {
  background: #ff4444;
  color: white;
}

.clear-btn:hover {
  background: #ff3333;
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.clear-btn:active {
  background: #e62e2e;
  transform: translateY(0px) scale(0.97);
  box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.predict-btn {
  background: #4CAF50;
  color: white;
}

.predict-btn:hover {
  background: #45a049;
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.predict-btn:active {
  background: #3d8b40;
  transform: translateY(0px) scale(0.97);
  box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.prediction-wrapper {
  flex: 1 1 50%;
  background: white;
  padding: 2rem;
  border-radius: 16px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.prediction-header {
  margin-bottom: 1.5rem;
}

.prediction-header h3 {
  color: #333;
  font-size: 1.4rem;
  margin: 0 0 1rem 0;
}

.confidence-meter {
  height: 8px;
  background: #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
}

.meter-bar {
  height: 100%;
  background: linear-gradient(90deg, #4CAF50, #45a049);
  transition: width 0.3s ease;
}

.result-container {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
  min-height: 100px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.result-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.prediction-text {
  font-size: 2.5rem;
  font-weight: bold;
  color: #333;
}

.confidence-text {
  font-size: 1.2rem;
  color: #666;
}

.placeholder {
  color: #666;
  font-style: italic;
}

.fade-out-enter-active, .fade-out-leave-active {
  transition: opacity 0.3s ease-out;
}
.fade-out-enter-from, .fade-out-leave-to {
  opacity: 0;
}
</style>
