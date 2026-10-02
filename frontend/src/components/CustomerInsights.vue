<script setup>
import { computed } from 'vue'
import { ArrowUpRight, BadgeCheck, BarChart3, Layers3, Sparkles, TrendingUp } from '@lucide/vue'

const props = defineProps({
  summary: { type: Object, default: null },
  recommendations: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
  aiExplanation: { type: Object, default: null },
  aiLoading: { type: Boolean, default: false },
})

defineEmits(['explain'])

const reasonLabels = {
  CUSTOMER_FREQUENT: 'Frequent',
  CUSTOMER_RECENT: 'Recent',
  CUSTOMER_REORDER: 'Reordered',
  DEPARTMENT_AFFINITY: 'Department Match',
  AISLE_AFFINITY: 'Aisle Match',
  GLOBAL_POPULARITY: 'Popular',
}

const reasonOrder = Object.keys(reasonLabels)

const purchaseMetrics = computed(() => {
  if (!props.summary) return []

  return [
    { label: 'Total orders', value: props.summary.total_orders?.toLocaleString?.() ?? '-' },
    { label: 'Products purchased', value: props.summary.products_purchased?.toLocaleString?.() ?? '-' },
    { label: 'Most active department', value: props.summary.most_active_department_id == null ? '-' : `#${props.summary.most_active_department_id}` },
    { label: 'Recent order', value: props.summary.recent_order_number == null ? '-' : `#${props.summary.recent_order_number}` },
  ]
})

const breakdown = computed(() => {
  const counts = Object.fromEntries(reasonOrder.map((code) => [code, 0]))

  for (const recommendation of props.recommendations) {
    const codes = Array.isArray(recommendation.explanation_reason_codes) ? recommendation.explanation_reason_codes : []
    for (const code of codes) {
      if (Object.hasOwn(counts, code)) {
        counts[code] += 1
      }
    }
  }

  const total = Object.values(counts).reduce((sum, value) => sum + value, 0)
  const maxCount = Math.max(1, ...Object.values(counts))

  return reasonOrder.map((code) => ({
    code,
    label: reasonLabels[code],
    count: counts[code],
    width: total === 0 ? 0 : Math.max(8, (counts[code] / maxCount) * 100),
  }))
})

const topRecommendation = computed(() => {
  if (!props.recommendations.length) return null
  return props.recommendations.reduce((winner, current) => {
    if (!winner || Number(current.model_score) > Number(winner.model_score)) {
      return current
    }
    return winner
  }, props.recommendations[0])
})

const scoreStats = computed(() => {
  if (!props.recommendations.length) {
    return { highest: 0, average: 0, count: 0 }
  }

  const values = props.recommendations.map((item) => Number(item.model_score) || 0)
  const highest = Math.max(...values)
  const average = values.reduce((sum, value) => sum + value, 0) / values.length

  return {
    highest,
    average,
    count: values.length,
  }
})

const insightNarrative = computed(() => {
  if (!topRecommendation.value) return 'No recommendation evidence is available yet.'

  const messages = []
  const codes = new Set(topRecommendation.value.explanation_reason_codes || [])

  if (codes.has('CUSTOMER_FREQUENT')) messages.push('Frequently purchased by this customer.')
  if (codes.has('CUSTOMER_RECENT')) messages.push('Purchased recently by this customer.')
  if (codes.has('CUSTOMER_REORDER')) messages.push('Previously reordered by this customer.')
  if (codes.has('DEPARTMENT_AFFINITY')) messages.push('Matches a department this customer frequently shops in.')
  if (codes.has('AISLE_AFFINITY')) messages.push('Matches an aisle from the customer\'s purchase history.')
  if (codes.has('GLOBAL_POPULARITY')) messages.push('Has purchase history across customers.')

  return messages.length ? messages.join(' ') : 'This recommendation is supported by the customer\'s historical buying patterns.'
})

function scorePercent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`
}
</script>

<template>
  <section
    v-if="!error && (loading || recommendations.length || summary)"
    class="insights-section"
    aria-labelledby="insights-heading"
  >
    <div class="section-heading insights-heading">
      <div>
        <p class="eyebrow">Customer insights</p>
        <h2 id="insights-heading">Why these recommendations fit this customer</h2>
      </div>
    </div>

    <div v-if="loading" class="insights-loading">
      <span class="loading-spinner" aria-hidden="true"></span>
      <span>Preparing customer insights...</span>
    </div>

    <div v-else-if="!recommendations.length" class="insights-empty">
      <span class="empty-state-mark">—</span>
      <h3>No insights yet</h3>
      <p>Generate a recommendation shortlist to view the customer evidence behind the ranking.</p>
    </div>

    <div v-else class="insights-grid">
      <article class="insight-panel purchase-panel">
        <div class="insight-header">
          <span class="insight-icon"><TrendingUp :size="16" :stroke-width="1.8" /></span>
          <h3>Purchase behavior</h3>
        </div>
        <ul class="purchase-metrics">
          <li v-for="metric in purchaseMetrics" :key="metric.label">
            <span>{{ metric.label }}</span>
            <strong>{{ metric.value }}</strong>
          </li>
        </ul>
      </article>

      <article class="insight-panel reason-panel">
        <div class="insight-header">
          <span class="insight-icon"><BarChart3 :size="16" :stroke-width="1.8" /></span>
          <h3>Reason breakdown</h3>
        </div>
        <div class="reason-breakdown">
          <div v-for="item in breakdown" :key="item.code" class="reason-row">
            <div class="reason-meta">
              <span>{{ item.label }}</span>
              <strong>{{ item.count }}</strong>
            </div>
            <div class="reason-bar" aria-hidden="true">
              <span :style="{ width: `${item.width}%` }"></span>
            </div>
          </div>
        </div>
      </article>

      <article class="insight-panel score-panel">
        <div class="insight-header">
          <span class="insight-icon"><Sparkles :size="16" :stroke-width="1.8" /></span>
          <h3>Recommendation score insight</h3>
        </div>
        <div class="score-grid">
          <div class="score-mini">
            <span>Highest</span>
            <strong>{{ scorePercent(scoreStats.highest) }}</strong>
          </div>
          <div class="score-mini">
            <span>Average</span>
            <strong>{{ scorePercent(scoreStats.average) }}</strong>
          </div>
          <div class="score-mini">
            <span>Displayed</span>
            <strong>{{ scoreStats.count }}</strong>
          </div>
        </div>
      </article>

      <article class="insight-panel highlight-panel">
        <div class="insight-header">
          <span class="insight-icon"><Layers3 :size="16" :stroke-width="1.8" /></span>
          <h3>Top recommendation</h3>
        </div>

        <div v-if="topRecommendation" class="top-recommendation">
          <div class="top-recommendation-header">
            <span class="rank-pill">#{{ topRecommendation.rank }}</span>
            <span class="score-pill">{{ scorePercent(topRecommendation.model_score) }}</span>
          </div>
          <h4>{{ topRecommendation.product_name }}</h4>
          <p>Department #{{ topRecommendation.department_id }} | Aisle #{{ topRecommendation.aisle_id }}</p>
          <div class="reason-badges compact-badges">
            <span
              v-for="code in topRecommendation.explanation_reason_codes"
              :key="`${topRecommendation.product_id}-${code}`"
              class="reason-badge"
            >
              <BadgeCheck :size="12" :stroke-width="2" />
              {{ reasonLabels[code] || code.replaceAll('_', ' ').toLowerCase() }}
            </span>
          </div>
          <button
            class="explain-button"
            type="button"
            :disabled="aiLoading"
            @click="$emit('explain', topRecommendation)"
          >
            <span v-if="aiLoading" class="button-spinner" aria-hidden="true"></span>
            <Sparkles v-else :size="14" :stroke-width="1.8" aria-hidden="true" />
            {{ aiLoading ? 'Preparing explanation...' : 'Explain this recommendation' }}
          </button>
          <div v-if="aiLoading" class="ai-explanation-state" role="status" aria-live="polite">
            <span class="loading-spinner" aria-hidden="true"></span>
            <span>Preparing an explanation from verified evidence...</span>
          </div>
          <div v-else-if="aiExplanation" class="ai-explanation-state" aria-live="polite">
            <strong>AI Explanation</strong>
            <p v-if="aiExplanation.available">{{ aiExplanation.ai_explanation }}</p>
            <p v-else>{{ aiExplanation.message || 'AI explanation is currently unavailable.' }}</p>
            <small v-if="aiExplanation.available">Generated from verified recommendation evidence.</small>
          </div>
        </div>
      </article>

      <article class="insight-panel explanation-panel">
        <div class="insight-header">
          <span class="insight-icon"><ArrowUpRight :size="16" :stroke-width="1.8" /></span>
          <h3>Why this recommendation?</h3>
        </div>
        <p class="insight-explanation">{{ insightNarrative }}</p>
      </article>
    </div>
  </section>
</template>
