<script setup>
import { Activity, CircleUserRound, Sparkles } from '@lucide/vue'

defineProps({
  backendHealthy: { type: Boolean, default: false },
  customerId: { type: [Number, String], default: 1 },
})
</script>

<template>
  <header class="app-header">
    <a class="brand" href="#top" aria-label="DecisionPilot home">
      <span class="brand-mark"><Sparkles :size="19" :stroke-width="2.2" /></span>
      <span class="brand-copy">
        <strong>DecisionPilot</strong>
        <small>AI-Powered Next Basket Prediction</small>
      </span>
    </a>

    <div class="header-meta">
      <span class="service-status" :class="{ 'is-online': backendHealthy }">
        <Activity :size="15" :stroke-width="2" />
        <span>{{ backendHealthy ? 'Backend connected' : 'Backend checking' }}</span>
      </span>
      <span class="customer-indicator">Customer #{{ customerId }}</span>
      <span class="header-divider" aria-hidden="true"></span>
      <span class="profile-mark" aria-label="Demo workspace">
        <CircleUserRound :size="19" :stroke-width="1.8" />
      </span>
    </div>
  </header>
</template>