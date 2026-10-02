<script setup>
import { Boxes, Layers3, ShoppingBasket, TrendingUp } from '@lucide/vue'
import LoadingState from './LoadingState.vue'

defineProps({
  summary: { type: Object, default: null },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
})

const metrics = [
  { key: 'total_orders', label: 'Total orders', icon: ShoppingBasket, format: (value) => value.toLocaleString() },
  { key: 'products_purchased', label: 'Products purchased', icon: Boxes, format: (value) => value.toLocaleString() },
  { key: 'most_active_department_id', label: 'Most active department', icon: Layers3, format: (value) => value === null ? '-' : `#${value}` },
  { key: 'recent_order_number', label: 'Recent order', icon: TrendingUp, format: (value) => value === null ? '-' : `#${value}` },
]
</script>

<template>
  <section class="summary-section" aria-labelledby="summary-heading">
    <div class="section-heading summary-heading">
      <div>
        <p class="eyebrow">Customer snapshot</p>
        <h2 id="summary-heading">Purchase history</h2>
      </div>
      <span v-if="summary" class="customer-chip">Customer {{ summary.customer_id }}</span>
    </div>

    <LoadingState v-if="loading && !summary" compact label="Loading customer history..." />
    <p v-else-if="error && !summary" class="inline-error">{{ error }}</p>
    <div v-else-if="summary" class="summary-grid">
      <article v-for="metric in metrics" :key="metric.key" class="summary-card">
        <div class="summary-card-top">
          <span class="summary-icon"><component :is="metric.icon" :size="17" :stroke-width="1.8" /></span>
          <span class="summary-label">{{ metric.label }}</span>
        </div>
        <strong class="summary-value">{{ metric.format(summary[metric.key]) }}</strong>
      </article>
    </div>
    <p v-else class="summary-empty">Customer history will appear here after a successful request.</p>
  </section>
</template>