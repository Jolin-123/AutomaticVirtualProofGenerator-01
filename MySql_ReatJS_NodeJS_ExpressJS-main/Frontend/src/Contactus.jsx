import React from 'react'
import './webpagestyle.css'
import { Link } from 'react-router-dom'
import letUsWorkForYou from './images/letUsWorkForYou.png';
import aboutus301 from './images/aboutus301.png';
import aboutus022 from './images/aboutus022.png';
import aboutus023 from './images/aboutus023.png';

function Contactus() {
  return (
    <div>
        
    <div>
      <header>
        <button className="menu">&#9776;</button>
        <h1>Products</h1>
        <nav>
          <ul>
            <li><Link to="/Home">Home</Link></li>
            <li><Link to="/NewProduct">Product</Link></li>
            <li><Link to="/Contactus">Contact Us</Link></li>
            <li><Link to="/Login">Login</Link></li>
          </ul>
        </nav>
      </header>

      <main>
        <section className="aboutus newsLayer">

        </section>

        <section className="slogan">
          <blockquote>"Carefully selected premium materials, crafted with ingenuity, allowing fashion to flow through every T-shirt."</blockquote>
        </section>

        <section className="shop">
          {/* <img src="image/lacityCloth.png" alt="烘焙工坊大安店" /> */}
          <img src = {letUsWorkForYou} alt = "letUsWorkForYou"/>
          <div className="info">
            <p>
            <h2>Our Mission </h2>
               <p>  Our mission is to empower individuals to express their unique
            identities through fashion. We strive to provide high-quality, ethically 
            sourced clothing.  </p>
            <p> <strong> Email : </strong> milklinjob@gmail.com </p>
            <p> <strong> Phone : </strong> 123-345-6789</p>
            <p> <strong> Location: </strong>  Downtown LA </p>
            </p>
          </div>
        </section>

        <section className="product">
          <div>
            {/* <img src="image/sewing.png" alt="sewing" /> */}
            <img src = {aboutus301} alt = "aboutus301"/>
            <h3>Design Logo </h3>
            <p>Create memorable visual identity reflecting brand values and personality.</p>
          </div>

          <div>
            {/* <img src="image/sewingThread.png" alt="Sewing Thread" /> */}
            <img src = {aboutus022} alt = "aboutus022"/>
            <h3> Pick Up Style </h3>
            <p>Showcase individuality through carefully curated fashion choices.</p>
          </div>

          <div>
            {/* <img src="image/Thread.png" alt="Customize Color" /> */}
            <img src = {aboutus023} alt = "aboutus023"/>
            <h3> Dye Fabric </h3>
            <p>Transform textiles with vibrant colors, adding unique flair to garments and home decor items.</p>
          </div>
        </section>

        <section className="newsletter">
          <p>Subscribe to our fashion newsletter, regularly receive the latest styles and special offers!</p>
          <form>
            <input type="email" placeholder="Please enter your Email address" required />
            <button type="submit">Submit</button>
          </form>
        </section>
      </main>

      <footer>
        <p>Copyright © 2023 Fashion T-Shirt Manufacturing Company All Right Reserved</p>
      </footer>
    </div>

    </div>
  )
}

export default Contactus