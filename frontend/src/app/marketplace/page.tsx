'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { useAppStore } from '@/store/appStore'
import { apiClient } from '@/services/api'
import { Product } from '@/types'
import { FiSearch, FiFilter, FiShoppingCart, FiAlertCircle } from 'react-icons/fi'

export default function MarketplacePage() {
  const { user } = useAppStore()
  const [products, setProducts] = useState<Product[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedType, setSelectedType] = useState('')

  useEffect(() => {
    fetchProducts()
  }, [selectedType])

  const fetchProducts = async () => {
    setLoading(true)
    setError('')
    try {
      const response = await apiClient.searchProducts({
        product_type: selectedType || undefined,
        limit: 20,
      })

      if (response.status === 'success' && response.data?.products) {
        setProducts(response.data.products)
      } else {
        setError('Failed to load products')
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load products')
    } finally {
      setLoading(false)
    }
  }

  const filteredProducts = products.filter((product) =>
    product.name.toLowerCase().includes(searchTerm.toLowerCase())
  )

  if (!user) {
    return (
      <div className="container-custom py-12">
        <div className="card text-center">
          <h2 className="text-2xl font-bold mb-4">Please Login</h2>
          <p className="text-gray-600 mb-6">You need to login to access the marketplace.</p>
          <Link href="/login" className="btn-primary inline-block">
            Go to Login
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="container-custom py-12">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Marketplace</h1>
        <p className="text-gray-600">Browse and purchase fresh agricultural products</p>
      </div>

      {/* Search and Filter */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="md:col-span-2 relative">
          <FiSearch className="absolute left-3 top-3 text-gray-400" />
          <input
            type="text"
            placeholder="Search products..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
          />
        </div>
        <div className="relative">
          <FiFilter className="absolute left-3 top-3 text-gray-400" />
          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
          >
            <option value="">All Products</option>
            <option value="vegetables">Vegetables</option>
            <option value="grains">Grains</option>
            <option value="pulses">Pulses</option>
            <option value="dairy">Dairy</option>
            <option value="honey">Honey</option>
            <option value="spices">Spices</option>
          </select>
        </div>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
          <FiAlertCircle className="text-red-600 mt-0.5 flex-shrink-0" />
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}

      {loading ? (
        <div className="text-center py-12">
          <p className="text-gray-600">Loading products...</p>
        </div>
      ) : filteredProducts.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-gray-600">No products found</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredProducts.map((product) => (
            <div key={product.product_id} className="card hover:shadow-lg transition-shadow">
              <div className="mb-4">
                <div className="w-full h-40 bg-gray-200 rounded-lg flex items-center justify-center">
                  {product.images && product.images.length > 0 ? (
                    <img
                      src={product.images[0]}
                      alt={product.name}
                      className="w-full h-full object-cover rounded-lg"
                    />
                  ) : (
                    <span className="text-gray-400">No image</span>
                  )}
                </div>
              </div>

              <h3 className="text-lg font-bold mb-2">{product.name}</h3>
              <p className="text-gray-600 text-sm mb-2 capitalize">{product.product_type}</p>

              <div className="grid grid-cols-2 gap-2 mb-4 text-sm">
                <div>
                  <p className="text-gray-600">Quantity</p>
                  <p className="font-bold">{product.quantity_kg} kg</p>
                </div>
                <div>
                  <p className="text-gray-600">Price</p>
                  <p className="font-bold">₹{product.price_per_kg}/kg</p>
                </div>
              </div>

              <div className="mb-4 p-2 bg-green-50 rounded">
                <p className="text-xs text-gray-600">Quality Score</p>
                <p className="text-lg font-bold text-green-600">{(product.pure_product_score * 100).toFixed(1)}%</p>
              </div>

              <div className="flex gap-2">
                <button className="flex-1 btn-primary flex items-center justify-center gap-2">
                  <FiShoppingCart />
                  Add to Cart
                </button>
                <button className="flex-1 btn-outline">View Details</button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
