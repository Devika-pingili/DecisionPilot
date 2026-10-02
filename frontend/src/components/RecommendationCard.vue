<script setup>
import { ArrowUpRight, BadgeCheck } from '@lucide/vue'

defineProps({
  recommendation: { type: Object, required: true },
})

const reasonLabels = {
  CUSTOMER_FREQUENT: 'Frequent',
  CUSTOMER_RECENT: 'Recent',
  CUSTOMER_REORDER: 'Reordered',
  DEPARTMENT_AFFINITY: 'Department Match',
  AISLE_AFFINITY: 'Aisle Match',
  GLOBAL_POPULARITY: 'Popular',
}

function scorePercent(score) {
  return `${(Number(score) * 100).toFixed(2)}%`
}
</script>

<template>
  <article class="recommendation-card">
    <div class="recommendation-topline">
      <span class="rank-label">RANK <strong>#{{ recommendation.rank }}</strong></span>
      <span class="score-label">Recommendation score</span>
    </div>

    <div class="recommendation-main">
      <div class="product-monogram" aria-hidden="true">{{ recommendation.product_name.slice(0, 1).toUpperCase() }}</div>
      <div class="product-copy">
        <h3>{{ recommendation.product_name }}</h3>
        <p>Department #{{ recommendation.department_id }} <span aria-hidden="true">|</span> Aisle #{{ recommendation.aisle_id }}</p>
      </div>
      <div class="score-value">{{ scorePercent(recommendation.model_score) }}</div>
    </div>

    <div class="score-track" aria-hidden="true">
      <span :style="{ width: `${Math.max(0, Math.min(100, Number(recommendation.model_score) * 100))}%` }"></span>
    </div>

    <p class="recommendation-explanation">{{ recommendation.explanation_short }}</p>

    <div class="reason-footer">
      <div class="reason-badges" aria-label="Historical evidence">
        <span
          v-for="code in recommendation.explanation_reason_codes"
          :key="code"
          class="reason-badge"
          :title="code"
        >
          <BadgeCheck :size="13" :stroke-width="2" />
          {{ reasonLabels[code] || code.replaceAll('_', ' ').toLowerCase() }}
        </span>
        <span v-if="recommendation.explanation_reason_codes.length === 0" class="reason-muted">No strong historical signals</span>
      </div>
      <ArrowUpRight class="card-arrow" :size="17" :stroke-width="1.8" aria-hidden="true" />
    </div>
  </article>
</template>