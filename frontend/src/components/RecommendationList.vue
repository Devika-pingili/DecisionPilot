<script setup>
import RecommendationCard from './RecommendationCard.vue'
import LoadingState from './LoadingState.vue'
import ErrorState from './ErrorState.vue'

defineProps({
  recommendations: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
})

defineEmits(['retry'])
</script>

<template>
  <div class="recommendation-content" aria-live="polite">
    <LoadingState v-if="loading" label="Analyzing purchase history..." />
    <ErrorState v-else-if="error" :message="error" @retry="$emit('retry')" />
    <div v-else-if="recommendations.length" class="recommendation-grid">
      <RecommendationCard
        v-for="recommendation in recommendations"
        :key="`${recommendation.rank}-${recommendation.product_id}`"
        :recommendation="recommendation"
      />
    </div>
    <div v-else class="empty-state">
      <span class="empty-state-mark">-</span>
      <h3>No recommendations yet</h3>
      <p>Choose a customer and generate a ranked shortlist to get started.</p>
    </div>
  </div>
</template>