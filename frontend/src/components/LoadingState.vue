<script setup>
import { LoaderCircle } from '@lucide/vue'

defineProps({
  label: { type: String, default: 'Loading...' },
  compact: { type: Boolean, default: false },
})
</script>

<template>
  <div class="loading-state" :class="{ 'is-compact': compact }" role="status">
    <LoaderCircle class="loading-spinner" :size="compact ? 18 : 22" :stroke-width="2" />
    <span>{{ label }}</span>
  </div>
</template>