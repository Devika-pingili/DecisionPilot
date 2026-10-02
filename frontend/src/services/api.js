const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: { Accept: 'application/json', ...options.headers },
    })
  } catch {
    throw new Error('Unable to reach DecisionPilot. Please make sure the backend is running.')
  }

  if (!response.ok) {
    let detail = ''
    try {
      detail = (await response.json()).detail || ''
    } catch {
      detail = ''
    }

    if (response.status === 404) {
      throw new Error('That customer ID could not be found. Check the ID and try again.')
    }
    if (response.status === 400 && detail.toLowerCase().includes('insufficient')) {
      throw new Error('There is not enough order history for this customer yet.')
    }
    if (response.status === 400 && detail.toLowerCase().includes('order_number')) {
      throw new Error('That order number is not available for this customer. Choose an earlier prior order.')
    }
    if (response.status === 422) {
      throw new Error('Check the customer ID, order number, and Top K values, then try again.')
    }
    if (response.status === 503) {
      throw new Error('Recommendation service is not ready. Please check the backend and try again.')
    }
    throw new Error('Unable to load recommendations. Please try again in a moment.')
  }

  return response.json()
}

export function getHealth() {
  return request('/health')
}

export function getCustomerSummary(customerId) {
  return request(`/customers/${encodeURIComponent(customerId)}/summary`)
}

export function getRecommendations(customerId, topK = 5, orderNumber = null) {
  const query = new URLSearchParams({ top_k: String(topK) })
  if (orderNumber !== null && orderNumber !== undefined && orderNumber !== '') {
    query.set('order_number', String(orderNumber))
  }
  return request(`/customers/${encodeURIComponent(customerId)}/recommendations?${query}`)
}

export function explainRecommendation(customerId, productId, topK = 5, orderNumber = null) {
  return request(`/customers/${encodeURIComponent(customerId)}/recommendations/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ product_id: productId, top_k: topK, order_number: orderNumber }),
  })
}