<script setup>
import { AlertCircle, RotateCw } from '@lucide/vue'

defineProps({
  message: { type: String, required: true },
})

defineEmits(['retry'])
</script>

<template>
  <div class="error-state" role="alert">
    <span class="error-icon"><AlertCircle :size="20" :stroke-width="1.8" /></span>
    <div class="error-copy">
      <strong>Recommendations unavailable</strong>
      <p>{{ message }}</p>
    </div>
    <button class="retry-button" type="button" aria-label="Try again" @click="$emit('retry')">
      <RotateCw :size="16" :stroke-width="1.9" />
      <span>Retry</span>
    </button>
  </div>
</template>