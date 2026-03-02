import { Product } from '@/types'
import { FiShoppingCart, FiStar } from 'react-icons/fi'

interface ProductCardProps {
  product: Product
  onAddToCart?: (product: Product) => void
}

export const ProductCard: React.FC<ProductCardProps> = ({ product, onAddToCart }) => {
  return (
    <div className="card hover:shadow-lg transition-shadow">
      <div className="mb-4">
        <div className="w-full h-40 bg-gray-200 rounded-lg flex items-center justify-center overflow-hidden">
          {product.images && product.images.length > 0 ? (
            <img
              src={product.images[0]}
              alt={product.name}
              className="w-full h-full object-cover"
            />
          ) : (
            <span className="text-gray-400">No image</span>
          )}
        </div>
      </div>

      <h3 className="text-lg font-bold mb-1">{product.name}</h3>
      <p className="text-gray-600 text-sm mb-3 capitalize">{product.product_type}</p>

      <div className="grid grid-cols-2 gap-3 mb-4 text-sm">
        <div className="p-2 bg-gray-50 rounded">
          <p className="text-gray-600 text-xs">Quantity</p>
          <p className="font-bold">{product.quantity_kg} kg</p>
        </div>
        <div className="p-2 bg-gray-50 rounded">
          <p className="text-gray-600 text-xs">Price/kg</p>
          <p className="font-bold">₹{product.price_per_kg}</p>
        </div>
      </div>

      <div className="mb-4 p-3 bg-green-50 rounded flex items-center justify-between">
        <div>
          <p className="text-xs text-gray-600">Quality Score</p>
          <p className="text-lg font-bold text-green-600">{(product.pure_product_score * 100).toFixed(1)}%</p>
        </div>
        <FiStar className="text-yellow-400 text-2xl" />
      </div>

      <div className="flex gap-2">
        <button
          onClick={() => onAddToCart?.(product)}
          className="flex-1 btn-primary flex items-center justify-center gap-2 text-sm"
        >
          <FiShoppingCart />
          Add to Cart
        </button>
        <button className="flex-1 btn-outline text-sm">Details</button>
      </div>

      <p className="text-xs text-gray-500 mt-3">
        Listed {new Date(product.created_at).toLocaleDateString()}
      </p>
    </div>
  )
}
